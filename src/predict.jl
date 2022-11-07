function predict!(pred::Matrix, tree::Tree{L,K,T}, X) where {L<:GradientRegression,K,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 :
            nid = nid << 1 + 1
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
    end
    return nothing
end

function predict!(pred::Matrix, tree::Tree{L,K,T}, X) where {L<:Logistic,K,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 :
            nid = nid << 1 + 1
        end
        @inbounds pred[1, i] = clamp(pred[1, i] + tree.pred[1, nid], -15, 15)
    end
    return nothing
end

function predict!(pred::Matrix, tree::Tree{L,K,T}, X) where {L<:MLE2P,K,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 :
            nid = nid << 1 + 1
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
        @inbounds pred[2, i] = max(-15, pred[2, i] + tree.pred[2, nid])
    end
    return nothing
end

"""
    predict!(pred::Matrix, tree::Tree, X)

Generic fallback to add predictions of `tree` to existing `pred` matrix.
"""
function predict!(pred::Matrix, tree::Tree{L,K,T}, X) where {L<:Softmax,K,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 :
            nid = nid << 1 + 1
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
        @views pred[:, i] .= max.(-15, pred[:, i] .- maximum(pred[:, i]))
    end
    return nothing
end

"""
    predict!(pred::Matrix, tree::Tree, X)

Generic fallback to add predictions of `tree` to existing `pred` matrix.
"""
function predict!(pred::Matrix, tree::Tree{L,K,T}, X) where {L,K,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 :
            nid = nid << 1 + 1
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
    end
    return nothing
end

"""
    predict(tree::Tree{L,K,T}, X::AbstractMatrix)

Prediction from a single tree - assign each observation to its final leaf.
"""
function predict(tree::Tree{L,K,T}, X::AbstractMatrix) where {L,K,T}
    pred = zeros(T, K, size(X, 1))
    predict!(pred, tree, X)
    return pred
end

"""
    predict(model::EvoTree, X::AbstractMatrix; ntree_limit = length(model.trees))

Predictions from an EvoTree model - sums the predictions from all trees composing the model.
Use `ntree_limit=N` to only predict with the first `N` trees.
"""
function predict(
    m::EvoTree{L,K,T},
    X::AbstractMatrix;
    ntree_limit = length(m.trees),
) where {L,K,T}
    pred = zeros(T, K, size(X, 1))
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    for i = 1:ntree_limit
        predict!(pred, m.trees[i], X)
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
    δ𝑤,
    𝑖,
) where {L<:GradientRegression,T}
    p[1, n] = -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end
function pred_scalar_cpu!(
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:GradientRegression,T}
    -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end

# prediction in Leaf - MLE2P
function pred_leaf_cpu!(p, n, ∑::Vector, params::EvoTypes{L,T}, δ𝑤, 𝑖) where {L<:MLE2P,T}
    p[1, n] = -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
    p[2, n] = -params.eta * ∑[2] / (∑[4] + params.lambda * ∑[5])
end
function pred_scalar_cpu!(∑::AbstractVector{T}, params::EvoTypes{L,T}) where {L<:MLE2P,T}
    -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    δ𝑤,
    𝑖,
) where {L<:MultiClassRegression,T}
    K = size(p, 1)
    @inbounds for k = 1:K
        p[k, n] = -params.eta * ∑[k] / (∑[k+K] + params.lambda * ∑[2*K+1])
    end
end

# prediction in Leaf - QuantileRegression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    δ𝑤,
    𝑖,
) where {L<:QuantileRegression,T}
    p[1, n] = params.eta * quantile(δ𝑤[2, 𝑖], params.alpha) / (1 + params.lambda)
end

# prediction in Leaf - L1Regression
function pred_leaf_cpu!(
    p,
    n,
    ∑::Vector,
    params::EvoTypes{L,T},
    δ𝑤,
    𝑖,
) where {L<:L1Regression,T}
    p[1, n] = params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
function pred_scalar_cpu!(
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:L1Regression,T}
    params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
