function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K) where {L<:GradientRegression,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
    end
    return nothing
end

function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K) where {L<:Logistic,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds pred[1, i] = clamp(pred[1, i] + tree.pred[1, nid], -15, 15)
    end
    return nothing
end

function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K) where {L<:GaussianRegression,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
        @inbounds pred[2, i] = max(-15, pred[2, i] + tree.pred[2, nid])
    end
    return nothing
end

"""
    predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K)

Generic fallback for adding preditions of `tree` to existing `pred` matrix.
"""
function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X, K) where {L,T}
    @inbounds @threads for i in axes(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        for k = 1:K
            @inbounds pred[k, i] += tree.pred[k, nid]
        end
    end
    return nothing
end

"""
    predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K)

Prediction from a single tree - assign each observation to its final leaf.
"""
function predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K) where {L,T}
    pred = zeros(T, K, size(X, 1))
    predict!(loss, pred, tree, X, K)
    return pred
end

"""
    predict(model::GBTree{T}, X::AbstractMatrix)

Predictions from an EvoTrees model - sums the predictions from all trees composing the model.
"""
function predict(model::GBTree{T}, X::AbstractMatrix) where {T}
    pred = zeros(T, model.K, size(X, 1))
    for tree in model.trees
        predict!(model.params.loss, pred, tree, X, model.K)
    end
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Gaussian
        pred[2, :] .= exp.(pred[2, :])
    elseif typeof(model.params.loss) == Softmax
        @inbounds for i in axes(pred, 2)
            pred[:, i] .= softmax(pred[:, i])
        end
    end
    return Array(transpose(pred))
end


function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes, K, δ𝑤, 𝑖) where {S<:GradientRegression,T}
    pred[1, n] = -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end
function pred_scalar_cpu!(::S, ∑::Vector{T}, params::EvoTypes, K) where {S<:GradientRegression,T}
    -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3])
end

# prediction in Leaf - GaussianRegression
function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes, K, δ𝑤, 𝑖) where {S<:GaussianRegression,T}
    pred[1, n] = -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
    pred[2, n] = -params.eta * ∑[2] / (∑[4] + params.lambda * ∑[5])
end
function pred_scalar_cpu!(::S, ∑::Vector{T}, params::EvoTypes, K) where {S<:GaussianRegression,T}
    -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5])
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes, K) where {S<:MultiClassRegression,T}
    @inbounds for k = 1:K
        pred[k, n] = -params.eta * ∑[k] / (∑[k+K] + params.lambda * ∑[2*K+1])
    end
end

# prediction in Leaf - QuantileRegression
function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes, K, δ𝑤, 𝑖) where {S<:QuantileRegression,T}
    pred[1, n] = params.eta * quantile(δ𝑤[2, 𝑖], params.alpha) / (1 + params.lambda)
    # pred[1,n] = params.eta * quantile(view(δ𝑤, 2, 𝑖), params.alpha) / (1 + params.lambda)
end
# function pred_scalar_cpu!(::S, ∑::Vector{T}, params::EvoTypes, K) where {S<:QuantileRegression,T}
#     params.eta * quantile(δ𝑤[2, 𝑖], params.alpha) / (1 + params.lambda)
# end

# prediction in Leaf - L1Regression
function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes, K, δ𝑤, 𝑖) where {S<:L1Regression,T}
    pred[1, n] = params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end
function pred_scalar_cpu!(::S, ∑::Vector{T}, params::EvoTypes, K) where {S<:L1Regression,T}
    params.eta * ∑[1] / (∑[3] * (1 + params.lambda))
end