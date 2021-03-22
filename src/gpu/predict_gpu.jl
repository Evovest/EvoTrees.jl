# prediction from single tree - assign each observation to its final leaf
function predict!(pred::AbstractMatrix{T}, tree::TreeGPU{T,S}, X::AbstractMatrix) where {T,S}
    @inbounds @threads for i in 1:size(X,1)
        K = length(tree.nodes[1].pred)
        id = 1
        x = view(X, i, :)
        @inbounds while tree.nodes[id].split
            if x[tree.nodes[id].feat] < tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        @inbounds for k in 1:K
            pred[i,k] += tree.nodes[id].pred[k]
        end
    end
end

# prediction from single tree - assign each observation to its final leaf
function predict(tree::TreeGPU{T,S}, X::AbstractMatrix, K) where {T,S}
    pred = zeros(T, size(X, 1), K)
    predict!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{T,S}, X::AbstractMatrix) where {T,S}
    K = length(model.trees[1].nodes[1].pred)
    pred = zeros(T, size(X, 1), K)
    for tree in model.trees
        predict!(pred, tree, X)
    end
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Gaussian
        pred[:,2] = exp.(pred[:,2])
    elseif typeof(model.params.loss) == Softmax
        pred = transpose(reshape(pred, model.K, :))
        for i in 1:size(pred,1)
            pred[i,:] .= softmax(pred[i,:])
        end
    end
    return pred
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu(::L, node::TrainNodeGPU{T}, params::EvoTypes) where {L<:GradientRegression,T}
    [- params.η * node.∑[1] / (node.∑[2] + params.λ * node.∑[3])]
end

# prediction in Leaf - GaussianRegression
function pred_leaf_gpu(::L, node::TrainNodeGPU{T}, params::EvoTypes) where {L<:GaussianRegression,T}
    [- params.η * node.∑[1] / (node.∑[3] + params.λ * node.∑[5]), - params.η * node.∑[2] / (node.∑[4] + params.λ * node.∑[5])]
end
