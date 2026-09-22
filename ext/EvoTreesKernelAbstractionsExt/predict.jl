# shared tree-walk: descend one observation to its leaf node (device helper)
@inline function _leaf_index_gpu(split, feats, cond_bins, x_bin, feattypes, i)
    nid = 1
    @inbounds while split[nid]
        feat = feats[nid]
        cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
        nid = (nid << 1) + Int(!cond)
    end
    return nid
end

# generic / GradientRegression / MLogLoss: plain accumulation
@kernel function predict_kernel!(::Type{L}, pred, @Const(split), @Const(feats), @Const(cond_bins), @Const(leaf_pred), @Const(x_bin), @Const(feattypes)) where {L}
    i = @index(Global, Linear)
    K = size(pred, 1)
    @inbounds if i <= size(pred, 2)
        nid = _leaf_index_gpu(split, feats, cond_bins, x_bin, feattypes, i)
        for k in 1:K
            pred[k, i] += leaf_pred[k, nid]
        end
    end
end

# LogLoss: clamp
@kernel function predict_kernel!(::Type{L}, pred, @Const(split), @Const(feats), @Const(cond_bins), @Const(leaf_pred), @Const(x_bin), @Const(feattypes)) where {L<:EvoTrees.LogLoss}
    i = @index(Global, Linear)
    T = eltype(pred)
    K = size(pred, 1)
    @inbounds if i <= size(pred, 2)
        nid = _leaf_index_gpu(split, feats, cond_bins, x_bin, feattypes, i)
        for k in 1:K
            pred[k, i] = clamp(pred[k, i] + leaf_pred[k, nid], T(-15), T(15))
        end
    end
end

# MLE2P: μ accumulates; unconstrained scale φ is clamped at a lower bound
@kernel function predict_kernel!(::Type{L}, pred, @Const(split), @Const(feats), @Const(cond_bins), @Const(leaf_pred), @Const(x_bin), @Const(feattypes)) where {L<:EvoTrees.MLE2P}
    i = @index(Global, Linear)
    T = eltype(pred)
    Y = size(pred, 1) ÷ 2
    @inbounds if i <= size(pred, 2)
        nid = _leaf_index_gpu(split, feats, cond_bins, x_bin, feattypes, i)
        for t in 1:Y
            pred[2t-1, i] += leaf_pred[2t-1, nid]
            pred[2t, i] = max(T(-15), pred[2t, i] + leaf_pred[2t, nid])
        end
    end
end

# leaf index extraction (loss-independent)
@kernel function leaf_index_kernel!(leaves, @Const(split), @Const(feats), @Const(cond_bins), @Const(x_bin), @Const(feattypes))
    i = @index(Global, Linear)
    @inbounds if i <= length(leaves)
        leaves[i] = _leaf_index_gpu(split, feats, cond_bins, x_bin, feattypes, i)
    end
end

# prediction from single tree
function EvoTrees.predict!(
    pred::CuMatrix{T},
    tree::EvoTrees.Tree{L,K},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
) where {L,K,T}
    backend = get_backend(pred)
    predict_kernel!(backend)(
        L, pred,
        _to_device(backend, tree.split), _to_device(backend, tree.feat),
        _to_device(backend, tree.cond_bin), _to_device(backend, tree.pred),
        x_bin, feattypes;
        ndrange=size(pred, 2),
    )
    KernelAbstractions.synchronize(backend)
end

function EvoTrees.predict!(
    pred::CuMatrix{T},
    tree::EvoTrees.Tree{L,K},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
) where {L<:EvoTrees.MLogLoss,K,T}
    backend = get_backend(pred)
    predict_kernel!(backend)(
        L, pred,
        _to_device(backend, tree.split), _to_device(backend, tree.feat),
        _to_device(backend, tree.cond_bin), _to_device(backend, tree.pred),
        x_bin, feattypes;
        ndrange=size(pred, 2),
    )
    KernelAbstractions.synchronize(backend)
    pred .= max.(T(-15), pred .- maximum(pred, dims=1))
end

# leaf index for one tree
function EvoTrees.predict_leaf_index!(
    leaves::CuVector,
    tree::EvoTrees.Tree,
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
)
    backend = get_backend(leaves)
    leaf_index_kernel!(backend)(
        leaves,
        _to_device(backend, tree.split), _to_device(backend, tree.feat),
        _to_device(backend, tree.cond_bin),
        x_bin, feattypes;
        ndrange=length(leaves),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# prediction for EvoTree model
function EvoTrees._predict(
    m::EvoTrees.EvoTree{L,K},
    data,
    device::Type{<:EvoTrees.GPU};
    ntree_limit=length(m.trees)) where {L,K}

    EvoTrees.Tables.istable(data) ? data = EvoTrees.Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    backend = _gpu_backend(device)
    x_bin = EvoTrees.binarize(device, data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = size(x_bin, 1)
    pred = KernelAbstractions.zeros(backend, Float32, K, nobs)
    pred .= _to_device(backend, m.bias)
    feattypes = _to_device(backend, m.info[:feattypes])
    for i in 1:ntree_limit
        EvoTrees.predict!(pred, m.trees[i], x_bin, feattypes)
    end
    EvoTrees.apply_prediction_link!(pred, L)
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end

# leaf indices for EvoTree model
function EvoTrees._predict_leaf_indices(
    m::EvoTrees.EvoTree,
    data,
    device::Type{<:EvoTrees.GPU};
    ntree_limit=length(m.trees))

    EvoTrees.Tables.istable(data) ? data = EvoTrees.Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    backend = _gpu_backend(device)
    x_bin = EvoTrees.binarize(device, data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = size(x_bin, 1)
    feattypes = _to_device(backend, m.info[:feattypes])
    leaves = Matrix{Int}(undef, nobs, ntree_limit)
    for t in 1:ntree_limit
        leaves_t = KernelAbstractions.zeros(backend, Int, nobs)
        EvoTrees.predict_leaf_index!(leaves_t, m.trees[t], x_bin, feattypes)
        leaves[:, t] = Array(leaves_t)
    end
    return leaves
end

@kernel function softmax_kernel!(p)
    i = @index(Global, Linear)
    T = eltype(p)
    K, nobs = size(p)
    @inbounds if i <= nobs
        isum = zero(T)
        for k in 1:K
            p[k, i] = exp(p[k, i])
            isum += p[k, i]
        end
        for k in 1:K
            p[k, i] /= isum
        end
    end
end

function EvoTrees.softmax!(p::CuMatrix{T}) where {T}
    backend = get_backend(p)
    softmax_kernel!(backend)(p; ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Quantile - special case where ∇ is passed as argument
function quantile_gpu(x::AnyCuVector, alpha)
    x_sort = sort(x)
    idx = ceil(Int, alpha * length(x_sort))
    return only(Array(view(x_sort, idx:idx)))
end

function EvoTrees.pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTrees.EvoTypes, ∇::CuMatrix, is) where {L<:EvoTrees.Quantile,T}
    ϵ = eps(T)
    K = size(p, 1)
    denom = 1 + params.lambda + params.L2 / ∑[end]
    @inbounds for k in 1:K
        p[k, n] = params.eta / params.bagging_size * quantile_gpu(view(∇, K + k, is), params.alpha) / max(ϵ, denom)
    end
end