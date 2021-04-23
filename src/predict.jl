function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X) where {L <: GradientRegression,T}
    @inbounds @threads for i in 1:size(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds pred[1,i] += tree.pred[1, nid]
    end
    return nothing
end

function predict!(::L, pred::Matrix{T}, tree::Tree{T}, X) where {L <: GaussianRegression,T}
    @inbounds @threads for i in 1:size(X, 1)
        nid = 1
        @inbounds while tree.split[nid]
            X[i, tree.feat[nid]] < tree.cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds pred[1,i] += tree.pred[1, nid]
        @inbounds pred[2,i] += tree.pred[2, nid]
    end
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K) where {L,T}
    pred = zeros(T, K, size(X, 1))
    predict!(loss, pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTree{T}, X::AbstractMatrix) where {T}
    pred = zeros(T, model.K, size(X, 1))
    # pred = zeros(SVector{model.K,T}, size(X, 1))
    for tree in model.trees
        predict!(model.params.loss, pred, tree, X)
    end
    # pred = reinterpret(T, pred)
    if typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Gaussian
        # pred = transpose(reshape(pred, 2, :))
        pred[2,:] .= exp.(pred[2,:])
    elseif typeof(model.params.loss) == Softmax
        # pred = transpose(reshape(pred, model.K, :))
        for i in 1:size(pred, 1)
            pred[:,i] .= softmax(pred[:,i])
        end
    end
    return Array(transpose(pred))
end


# prediction in Leaf - GradientRegression
# function pred_leaf_cpu(::S, ∑::Vector{T}, params::EvoTypes) where {S <: GradientRegression,T}
#     - params.η .* ∑[1] ./ (∑[2] .+ params.λ .* ∑[3])
# end

function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes) where {S <: GradientRegression,T}
    pred[1,n] = - params.η .* ∑[1] ./ (∑[2] .+ params.λ .* ∑[3])
end

# prediction in Leaf - GaussianRegression
function pred_leaf_cpu!(::S, pred, n, ∑::Vector{T}, params::EvoTypes) where {S <: GaussianRegression,T}
    pred[1,n] = - params.η .* ∑[1] ./ (∑[3] .+ params.λ .* ∑[5])
    pred[2,n] = - params.η .* ∑[2] ./ (∑[4] .+ params.λ .* ∑[5])
end

# prediction in Leaf - GradientRegression
function pred_leaf(::S, node::TrainNode{T}, params::EvoTypes, δ²) where {S <: GradientRegression,T}
    - params.η .* node.∑δ ./ (node.∑δ² .+ params.λ .* node.∑𝑤)
end

# prediction in Leaf - MultiClassRegression
function pred_leaf(::S, node::TrainNode{T}, params::EvoTypes, δ²) where {S <: MultiClassRegression,T}
    SVector{L,T}(-params.η .* node.∑δ ./ (node.∑δ² .+ params.λ .* node.∑𝑤[1]))
end

# prediction in Leaf - L1Regression
function pred_leaf(::S, node::TrainNode{T}, params::EvoTypes, δ²) where {S <: L1Regression,T}
    params.η .* node.∑δ ./ (node.∑𝑤 .* (1 .+ params.λ))
end

# prediction in Leaf - QuantileRegression
function pred_leaf(::S, node::TrainNode{T}, params::EvoTypes, δ²) where {S <: QuantileRegression,L,T}
    SVector{1,T}(params.η * quantile(reinterpret(Float32, δ²[node.𝑖]), params.α) / (1 + params.λ))
    # pred = params.η * quantile(δ²[collect(node.𝑖)], params.α) / (1 + params.λ)
end

# prediction in Leaf - GaussianRegression
# function pred_leaf(::S, node::TrainNode{T}, params::EvoTypes, δ²) where {S <: GaussianRegression,T}
#     - params.η * node.∑δ ./ (node.∑δ² .+ params.λ .* node.∑𝑤[1])
# end
