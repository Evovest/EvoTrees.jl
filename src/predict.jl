function predict!(pred::Matrix, tree::Tree{L,T}, X, K) where {L<:GradientRegression,T}
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

function predict!(pred::Matrix, tree::Tree{L,T}, X, K) where {L<:Logistic,T}
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

function predict!(pred::Matrix, tree::Tree{L,T}, X, K) where {L<:MLE2P,T}
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
    predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K)

Generic fallback to add predictions of `tree` to existing `pred` matrix.
"""
function predict!(pred::Matrix, tree::Tree{L,T}, X, K) where {L,T}
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
    predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K)

Prediction from a single tree - assign each observation to its final leaf.
"""
function predict(tree::Tree{L,T}, X::AbstractMatrix, K) where {L,T}
    pred = zeros(T, K, size(X, 1))
    predict!(pred, tree, X, K)
    return pred
end

"""
    predict(model::GBTree{T}, X::AbstractMatrix)

Predictions from an EvoTrees model - sums the predictions from all trees composing the model.
"""
function predict(model::GBTree{L,T,S}, X::AbstractMatrix) where {L,T,S}
    pred = zeros(T, model.K, size(X, 1))
    for tree in model.trees
        predict!(pred, tree, X, model.K)
    end
    if L == Logistic
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L in [GaussianDist, LogisticDist]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == Softmax
        @inbounds for i in axes(pred, 2)
            pred[:, i] .= softmax(pred[:, i])
        end
    end
    pred = model.K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end


function pred_leaf_cpu!(
    pred,
    n,
    ∑::Vector,
    params::EvoTypes{L,T,S},
    K,
    δ𝑤,
    𝑖,
) where {L<:GradientRegression,T,S}
    pred[1, n] = -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end
function pred_scalar_cpu!(
    ∑::Vector{T},
    params::EvoTypes,
    K,
) where {L<:GradientRegression,T,S}
    -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end

# prediction in Leaf - MLE2P
function pred_leaf_cpu!(
    pred,
    n,
    ∑::Vector,
    params::EvoTypes{L,T,S},
    K,
    δ𝑤,
    𝑖,
) where {L<:MLE2P,T,S}
    pred[1, n] = -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
    pred[2, n] = -params.eta * ∑[2] / (∑[4] + params.lambda * ∑[5])
end
function pred_scalar_cpu!(∑::Vector{T}, params::EvoTypes{L,T,S}, K) where {L<:MLE2P,T,S}
    -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(
    pred,
    n,
    ∑::Vector,
    params::EvoTypes{L,T,S},
    K,
    δ𝑤,
    𝑖,
) where {L<:MultiClassRegression,T,S}
    @inbounds for k = 1:K
        pred[k, n] = -params.eta * ∑[k] / (∑[k+K] + params.lambda * ∑[2*K+1])
    end
end

# prediction in Leaf - QuantileRegression
function pred_leaf_cpu!(
    pred,
    n,
    ∑::Vector,
    params::EvoTypes{L,T,S},
    K,
    δ𝑤,
    𝑖,
) where {L<:QuantileRegression,T,S}
    pred[1, n] = params.eta * quantile(δ𝑤[2, 𝑖], params.alpha) / (1 + params.lambda)
    # pred[1,n] = params.eta * quantile(view(δ𝑤, 2, 𝑖), params.alpha) / (1 + params.lambda)
end
# function pred_scalar_cpu!(::S, ∑::Vector{T}, params::EvoTypes, K) where {S<:QuantileRegression,T}
#     params.eta * quantile(δ𝑤[2, 𝑖], params.alpha) / (1 + params.lambda)
# end

# prediction in Leaf - L1Regression
function pred_leaf_cpu!(
    pred,
    n,
    ∑::Vector,
    params::EvoTypes{L,T,S},
    K,
    δ𝑤,
    𝑖,
) where {L<:L1Regression,T,S}
    pred[1, n] = params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
function pred_scalar_cpu!(∑::Vector, params::EvoTypes{L,T,S}, K) where {L<:L1Regression,T,S}
    params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end