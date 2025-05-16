function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:GradientRegression,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:LogLoss,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] = clamp(pred[1, i] + tree.pred[1, nid], T(-15), T(15))
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:MLE2P,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
        @inbounds pred[2, i] = max(T(-15), pred[2, i] + tree.pred[2, nid])
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:MLogLoss,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
        @views pred[:, i] .= max.(T(-15), pred[:, i] .- maximum(pred[:, i]))
    end
    return nothing
end

"""
    predict!(pred::Matrix, tree::Tree, X)

Generic fallback to add predictions of `tree` to existing `pred` matrix.
"""
function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
    end
    return nothing
end


"""
    predict(m::EvoTree, data; ntree_limit=length(m.trees), device=:cpu)

Predictions from an EvoTree model - sums the predictions from all trees composing the model.
Use `ntree_limit=N` to only predict with the first `N` trees.
"""
function predict(m::EvoTree, data; ntree_limit=length(m.trees), device=:cpu)
    @assert Symbol(device) ∈ [:cpu, :gpu]
    _device = Symbol(device) == :cpu ? CPU : GPU
    _predict(m, data, _device; ntree_limit)
end

function _predict(
    m::EvoTree{L,K},
    data,
    ::Type{<:CPU};
    ntree_limit=length(m.trees)) where {L,K}

    Tables.istable(data) ? data = Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    x_bin = binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = size(x_bin, 1)
    pred = zeros(Float32, K, nobs)
    for i = 1:ntree_limit
        predict!(pred, m.trees[i], x_bin, m.info[:feattypes])
    end
    if L == LogLoss
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L in [GaussianMLE, LogisticMLE]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == MLogLoss
        softmax!(pred)
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
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
    p[1, n] = -params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[2] + params.lambda * ∑[3] + params.L2))
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:GradientRegression,T}
    ϵ = eps(T)
    return -params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[2] + params.lambda * ∑[3] + params.L2))
end

# Cred predictions
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:Cred,T}
    p[1, n] = params.eta / params.bagging_size * ∑[1] / (∑[3] + params.L2)
    return nothing
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:Cred,T}
    return params.eta / params.bagging_size * ∑[1] / (∑[3] + params.L2)
end

# prediction in Leaf - MLE2P
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MLE2P,T}
    ϵ = eps(T)
    p[1, n] = -params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[5] + params.L2))
    p[2, n] = -params.eta / params.bagging_size * ∑[2] / max(ϵ, (∑[4] + params.lambda * ∑[5] + params.L2))
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
    p[1, n] = params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[3] + params.L2))
end
function pred_scalar(∑::AbstractVector{T}, ::Type{L}, params::EvoTypes) where {L<:MAE,T}
    ϵ = eps(T)
    return params.eta / params.bagging_size * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[3] + params.L2))
end

# Quantile
function pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTypes, ∇, is) where {L<:Quantile,T}
    ϵ = eps(T)
    p[1, n] = params.eta / params.bagging_size * quantile(view(∇, 2, is), params.alpha) / (1 + params.lambda + params.L2 / ∑[3])
end
