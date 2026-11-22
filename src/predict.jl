"""
    PREDICT_THREAD_MIN

Row count below which `predict!` walks the batch serially. Threading the row loop opens a
parallel region per tree, so a small batch pays that cost once per tree for a few rows of
work. Measured on 200 trees at depth 5 with 8 threads: serial is 61x faster at 1 row, 6.9x
at 64 and 1.9x at 256, while threading wins from roughly 1024.
"""
const PREDICT_THREAD_MIN = 512

"""
    _leaf_index(tree, x_bin, feattypes, i)

Index of the leaf that row `i` falls into. Numeric and ordered features compare with `<=`,
unordered categoricals with `==`.
"""
@inline function _leaf_index(tree::Tree, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int)
    nid = 1
    @inbounds while tree.split[nid]
        feat = tree.feat[nid]
        cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
        nid = nid << 1 + !cond
    end
    return nid
end

@inline function _predict_row!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int) where {L<:GradientRegression,K,T}
    nid = _leaf_index(tree, x_bin, feattypes, i)
    @inbounds for k in axes(pred, 1)
        pred[k, i] += tree.pred[k, nid]
    end
end

@inline function _predict_row!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int) where {L<:LogLoss,K,T}
    nid = _leaf_index(tree, x_bin, feattypes, i)
    @inbounds for k in axes(pred, 1)
        pred[k, i] = clamp(pred[k, i] + tree.pred[k, nid], T(-15), T(15))
    end
end

@inline function _predict_row!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int) where {L<:MLE2P,K,T}
    nid = _leaf_index(tree, x_bin, feattypes, i)
    Y = size(pred, 1) ÷ 2
    @inbounds for t in 1:Y
        pred[2t-1, i] += tree.pred[2t-1, nid]
        pred[2t, i] = max(T(-15), pred[2t, i] + tree.pred[2t, nid])
    end
end

@inline function _predict_row!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int) where {L<:MLogLoss,K,T}
    nid = _leaf_index(tree, x_bin, feattypes, i)
    @inbounds for k = 1:K
        pred[k, i] += tree.pred[k, nid]
    end
    @views pred[:, i] .= max.(T(-15), pred[:, i] .- maximum(pred[:, i]))
end

@inline function _predict_row!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}, i::Int) where {L,K,T}
    nid = _leaf_index(tree, x_bin, feattypes, i)
    @inbounds for k = 1:K
        pred[k, i] += tree.pred[k, nid]
    end
end

"""
    predict!(pred::Matrix, tree::Tree, x_bin, feattypes)

Add the predictions of `tree` to `pred`. The row loop is threaded only from
`PREDICT_THREAD_MIN` rows up, since the parallel region is opened once per tree.
"""
function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L,K,T}
    n = size(x_bin, 1)
    if n >= PREDICT_THREAD_MIN
        @threads for i = 1:n
            _predict_row!(pred, tree, x_bin, feattypes, i)
        end
    else
        for i = 1:n
            _predict_row!(pred, tree, x_bin, feattypes, i)
        end
    end
    return nothing
end

function predict_leaf_index! end

function _predict_leaf_indices end

"""
    predict(m::EvoTree, data; ntree_limit=length(m.trees), device=:cpu)

Predictions from an EvoTree model — the bias plus the sum of boosting trees.
Use `ntree_limit=N` to use the bias plus the first `N` trees (`N = 0` is bias only).
"""
function predict(m::EvoTree, data; ntree_limit=length(m.trees), device=:cpu)
    _device = device_type(device)
    check_features(data, m.info[:feature_names])
    _predict(m, data, _device; ntree_limit)
end

"""
    apply_prediction_link!(pred, ::Type{L})

Map unconstrained tree sums onto the loss's response scale, in place.
Shared by CPU and GPU `_predict` so inverse-link transforms cannot drift.
`MLogLoss` dispatches to the device-specific `softmax!`.
"""
function apply_prediction_link!(pred, ::Type{L}) where {L}
    if L == LogLoss
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L <: MLE2P
        pred[2:2:end, :] .= exp.(pred[2:2:end, :])
    elseif L == MLogLoss
        softmax!(pred)
    end
    return pred
end

"""
    PREDICT_BLOCK_ROWS

Most rows that `predict` bins and scores at once. Each block is binned on its own and goes
through every tree while its bins are still in cache, so the input is never binned as a whole.
"""
const PREDICT_BLOCK_ROWS = 2048

_nobs(data::AbstractMatrix) = size(data, 1)
_nobs(data) = length(Tables.getcolumn(data, 1))

_rows(data::AbstractMatrix, rows) = view(data, rows, :)
_rows(data, rows) = map(col -> view(col, rows), data)

# Where `binarize!` reads each block from. A table's features are grouped by column type, each
# group's block passed as a Dict of views refilled in place: `binarize!` looks columns up by name,
# which on a NamedTuple of views copies it whole each time. Edges are narrowed to a concrete type
# (a table model stores them as `Vector{Any}`) so `binarize!` does not dispatch per column.
_bin_source(X::AbstractMatrix, feature_names, edges) = (; X, feature_names, edges=identity.(edges))
function _bin_source(data, feature_names, edges)
    cols = [Tables.getcolumn(data, name) for name in feature_names]
    return map(unique(typeof.(cols))) do C
        js = findall(col -> col isa C, cols)
        (; js, names=feature_names[js], edges=identity.(edges[js]), cols=Vector{C}(cols[js]))
    end
end

_bin_buffers(::NamedTuple) = nothing
_bin_buffers(groups) = [Dict(name => view(col, 1:0) for (name, col) in zip(g.names, g.cols)) for g in groups]

_bin_block!(x_bin, src::NamedTuple, ::Nothing, rows) =
    binarize!(view(x_bin, 1:length(rows), :), view(src.X, rows, :); src.feature_names, src.edges)
function _bin_block!(x_bin, groups, bufs, rows)
    for (g, buf) in zip(groups, bufs)
        _bin_group!(x_bin, g, buf, rows)
    end
end
function _bin_group!(x_bin, g, buf, rows)
    for (name, col) in zip(g.names, g.cols)
        buf[name] = view(col, rows)
    end
    binarize!(view(x_bin, 1:length(rows), g.js), buf; feature_names=g.names, edges=g.edges)
end

# Bias plus the first `ntree_limit` trees for one block, written to `pred[rows, :]`. The block's
# bins and scores are the first `length(rows)` rows of `x_bin` and columns of `p`.
function _predict_block!(pred, p::Matrix{T}, m::EvoTree{L,K}, x_bin::Matrix{UInt8}, rows, feattypes::Vector{Bool}, ntree_limit) where {T,L,K}
    n = length(rows)
    pv = view(p, :, 1:n)
    pv .= m.bias
    for j in 1:ntree_limit
        tree = m.trees[j]
        for i in 1:n
            _predict_row!(p, tree, x_bin, feattypes, i)
        end
    end
    apply_prediction_link!(pv, L)
    pred[rows, :] .= transpose(pv)
    return nothing
end

# Blocks of PREDICT_THREAD_MIN to PREDICT_BLOCK_ROWS rows, one per thread when the batch allows.
# Each task takes a run of blocks and reuses its buffers across them.
function _predict_blocks!(pred::Matrix{T}, m::EvoTree{L,K}, data, feature_names, edges, feattypes, ntree_limit) where {T,L,K}
    nobs = size(pred, 1)
    bs = clamp(cld(nobs, nthreads()), PREDICT_THREAD_MIN, PREDICT_BLOCK_ROWS)
    nblocks = cld(nobs, bs)
    if nblocks <= 1
        _predict_block!(pred, Matrix{T}(undef, K, nobs), m, binarize(data; feature_names, edges), 1:nobs, feattypes, ntree_limit)
        return nothing
    end
    # bad input fails here, once, rather than in every task
    binarize(_rows(data, 1:0); feature_names, edges)
    src = _bin_source(data, feature_names, edges)
    ntasks = min(nthreads(), nblocks)
    per = cld(nblocks, ntasks)
    @threads for t in 1:ntasks
        x_bin = Matrix{UInt8}(undef, bs, length(feature_names))
        p = Matrix{T}(undef, K, bs)
        bufs = _bin_buffers(src)
        for b in (t-1)*per+1:min(t * per, nblocks)
            rows = (b-1)*bs+1:min(b * bs, nobs)
            _bin_block!(x_bin, src, bufs, rows)
            _predict_block!(pred, p, m, x_bin, rows, feattypes, ntree_limit)
        end
    end
    return nothing
end

function _predict(
    m::EvoTree{L,K},
    data,
    ::Type{<:CPU};
    ntree_limit=length(m.trees)) where {L,K}

    Tables.istable(data) ? data = Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    pred = Matrix{Float32}(undef, _nobs(data), K)
    _predict_blocks!(pred, m, data, m.info[:feature_names], m.info[:edges], m.info[:feattypes], ntree_limit)
    return K == 1 ? vec(pred) : pred
end

function predict_leaf_idx!(
    leaf_idx::AbstractVector,
    tree::Tree{L,K},
    x_bin::Matrix{UInt8},
    feattypes::Vector{Bool},
) where {L,K}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds leaf_idx[i] = nid
    end
    return nothing
end

"""
    predict_leaf_idx(m::EvoTree, data; ntree_limit=length(m.trees))

Return the index of the leaf into which each observation falls, for each boosting tree.

The result is a `Matrix{UInt32}` of size `(nobs, ntree_limit)`, where `[i, j]` is the index of
the leaf reached by observation `i` in tree `j`. Indices refer to the node numbering of
`m.trees[j]`: the root is `1`, and the children of node `n` are `2n` and `2n + 1`.
Use `ntree_limit=N` to only use the first `N` trees. The bias is not a tree, so `nrounds = 0`
returns an `(nobs, 0)` matrix.

Leaf indices are a categorical encoding of the partition of the feature space learned by the
model, and can be used as features for a downstream model.

```julia
leaf_idx = EvoTrees.predict_leaf_idx(m, x)
```
"""
function predict_leaf_idx(m::EvoTree, data; ntree_limit=length(m.trees))
    Tables.istable(data) ? data = Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    check_features(data, m.info[:feature_names])
    x_bin = binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = size(x_bin, 1)
    leaf_idx = zeros(UInt32, nobs, ntree_limit)
    feattypes = m.info[:feattypes]
    for j = 1:ntree_limit
        predict_leaf_idx!(view(leaf_idx, :, j), m.trees[j], x_bin, feattypes)
    end
    return leaf_idx
end

function softmax!(p::AbstractMatrix)
    @threads for i in axes(p, 2)
        _p = view(p, :, i)
        _p .= exp.(_p)
        isum = sum(_p)
        _p ./= isum
    end
    return nothing
end

# GradientRegression predictions
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:GradientRegression,T}
    ϵ = eps(T)
    K = size(p, 1)
    @inbounds for k in 1:K
        p[k, n] = -params.eta / params.bagging_size * ∑[k] / max(ϵ, (∑[k+K] + params.lambda * ∑[end] + params.L2))
    end
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:GradientRegression,T}
    ϵ = eps(T)
    return -params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[2] + params.lambda * ∑[3] + params.L2))
end

# Cred predictions
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:Cred,T}
    K = size(p, 1)
    w = ∑[end]
    @inbounds for k in 1:K
        p[k, n] = params.eta / params.bagging_size * ∑[k] / (w + params.L2)
    end
    return nothing
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:Cred,T}
    return params.eta / params.bagging_size * ∑[1] / (∑[3] + params.L2)
end

# prediction in Leaf - MLE2P
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MLE2P,T}
    ϵ = eps(T)
    Y = size(p, 1) ÷ 2
    wsum = ∑[end]
    @inbounds for t in 1:Y
        gb = 2 * (t - 1)
        hb = 2 * Y + 2 * (t - 1)
        p[2t-1, n] = -params.eta / params.bagging_size * ∑[gb+1] / max(ϵ, (∑[hb+1] + params.lambda * wsum + params.L2))
        p[2t, n] = -params.eta / params.bagging_size * ∑[gb+2] / max(ϵ, (∑[hb+2] + params.lambda * wsum + params.L2))
    end
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MLE2P,T}
    ϵ = eps(T)
    return -params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[5] + params.L2))
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MLogLoss,T}
    ϵ = eps(T)
    K = size(p, 1)
    @inbounds for k = axes(p, 1)
        p[k, n] = -params.eta / params.bagging_size * ∑[k] / max(ϵ, (∑[k+K] + params.lambda * ∑[end] + params.L2))
    end
end

# MAE
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MAE,T}
    ϵ = eps(T)
    K = size(p, 1)
    w = ∑[end]
    denom = max(ϵ, (w + params.lambda * w + params.L2))
    @inbounds for k in 1:K
        p[k, n] = params.eta / params.bagging_size * ∑[k] / denom
    end
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MAE,T}
    ϵ = eps(T)
    return params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[3] + params.L2))
end

# Quantile
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes, ∇, is) where {L<:Quantile,T}
    ϵ = eps(T)
    K = size(p, 1)
    denom = 1 + params.lambda + params.L2 / ∑[end]
    @inbounds for k in 1:K
        p[k, n] = params.eta / params.bagging_size * quantile(view(∇, K + k, is), params.alpha) / max(ϵ, denom)
    end
end

# MultiQuantile
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes, ∇, is) where {L<:MultiQuantile,T}
    K = length(params.alphas)
    denom = 1 + params.lambda + params.L2 / ∑[end]
    @inbounds for k in 1:K
        p[k, n] = params.eta / params.bagging_size * quantile(view(∇, K + k, is), params.alphas[k]) / denom
    end
end
