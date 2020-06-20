# prediction from single tree - assign each observation to its final leaf
function predict_gpu!(pred, tree::Tree_gpu, X::AbstractMatrix{T}) where {T<:Real}
    @inbounds @threads for i in 1:size(X,1)
        id = 1
        x = view(X, i, :)
        @inbounds while tree.nodes[id].split
            if x[tree.nodes[id].feat] < tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu(tree::Tree_gpu, X::AbstractMatrix{T}, K) where T<:Real
    pred = CUDA.zeros(size(X, 1))
    predict_gpu!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu(model::GBTree_gpu, X::AbstractMatrix{T}) where T<:Real
    pred = zeros(SVector{model.K,Float32}, size(X, 1))
    for tree in model.trees
        predict_gpu!(pred, tree, X)
    end
    pred = reinterpret(Float32, pred)
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Gaussian
        pred = transpose(reshape(pred, 2, :))
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
function pred_leaf_gpu(loss::S, node::TrainNode_gpu{T}, params::EvoTypes, δ²) where {S<:GradientRegression,T}
    - params.η * node.∑δ / (node.∑δ² + params.λ * node.∑𝑤)
end
