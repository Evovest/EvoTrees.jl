function predict!(pred::Matrix{T}, tree::Tree{L,K,T}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:GradientRegression,K,T}
    @inbounds @threads for i in axes(x_bin, 1)
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

function predict!(pred::Matrix{T}, tree::Tree{L,K,T}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:Logistic,K,T}
    @inbounds @threads for i in axes(x_bin, 1)
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

function predict!(pred::Matrix{T}, tree::Tree{L,K,T}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:MLE2P,K,T}
    @inbounds @threads for i in axes(x_bin, 1)
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

function predict!(pred::Matrix{T}, tree::Tree{L,K,T}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:Softmax,K,T}
    @inbounds @threads for i in axes(x_bin, 1)
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
function predict!(pred::Matrix{T}, tree::Tree{L,K,T}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L,K,T}
    @inbounds @threads for i in axes(x_bin, 1)
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
    predict(model::EvoTree, X::AbstractMatrix; ntree_limit = length(model.trees))

Predictions from an EvoTree model - sums the predictions from all trees composing the model.
Use `ntree_limit=N` to only predict with the first `N` trees.
"""
function predict(
    m::EvoTree{L,K,T},
    data,
    ::Type{<:Device}=CPU;
    ntree_limit=length(m.trees)) where {L,K,T}

    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    x_bin = binarize(data; fnames=m.info[:fnames], edges=m.info[:edges])
    nobs = Tables.istable(data) ? length(Tables.getcolumn(data, 1)) : size(data, 1)
    pred = zeros(T, K, nobs)
    for i = 1:ntree_limit
        predict!(pred, m.trees[i], x_bin, m.info[:feattypes])
    end
    if L == Logistic
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L in [GaussianMLE, LogisticMLE]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == Softmax
        @inbounds for i in axes(pred, 2)
            pred[:, i] .= softmax(pred[:, i])
        end
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end

function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    ∇,
    is,
) where {L<:GradientRegression,T}
    ϵ = eps(T)
    p[1, n] = -params.eta * ∑[1] / max(ϵ, (∑[2] + params.lambda * ∑[3]))
end
function pred_scalar(
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:GradientRegression,T}
    ϵ = eps(T)
    -params.eta * ∑[1] / max(ϵ, (∑[2] + params.lambda * ∑[3]))
end

# prediction in Leaf - MLE2P
function pred_leaf_cpu!(p, n, ∑::Vector, params::EvoTypes{L,T}, ∇, is) where {L<:MLE2P,T}
    ϵ = eps(T)
    p[1, n] = -params.eta * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[5]))
    p[2, n] = -params.eta * ∑[2] / max(ϵ, (∑[4] + params.lambda * ∑[5]))
end
function pred_scalar(∑::AbstractVector{T}, params::EvoTypes{L,T}) where {L<:MLE2P,T}
    ϵ = eps(T)
    -params.eta * ∑[1] / max(ϵ, (∑[3] + params.lambda * ∑[5]))
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    ∇,
    is,
) where {L<:MultiClassRegression,T}
    K = size(p, 1)
    ϵ = eps(T)
    @inbounds for k = axes(p, 1)
        p[k, n] = -params.eta * ∑[k] / max(ϵ, (∑[k+K] + params.lambda * ∑[end]))
    end
end

# prediction in Leaf - QuantileRegression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    ∇,
    is,
) where {L<:QuantileRegression,T}
    p[1, n] = params.eta * quantile(∇[2, is], params.alpha) / (1 + params.lambda)
end

# prediction in Leaf - L1Regression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    ∇,
    is,
) where {L<:L1Regression,T}
    p[1, n] = params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
function pred_scalar(
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:L1Regression,T}
    params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
