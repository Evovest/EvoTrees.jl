# # prediction from single tree - assign each observation to its final leaf
# function predict!(pred::AbstractMatrix{T}, tree::TreeGPU{T}, X::AbstractMatrix) where {T}
#     @inbounds @threads for i in 1:size(X, 1)
#         K = length(tree.nodes[1].pred)
#         id = 1
#         x = view(X, i, :)
#         @inbounds while tree.nodes[id].split
#             if x[tree.nodes[id].feat] < tree.nodes[id].cond
#                 id = tree.nodes[id].left
#             else
#                 id = tree.nodes[id].right
#             end
#         end
#         @inbounds for k in 1:K
#             pred[i,k] += tree.nodes[id].pred[k]
#         end
#     end
# end

function predict_kernel!(pred::AbstractMatrix{T}, tree::TreeGPU{T}, X::CuDeviceMatrix) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i <= length(size(pred, 1))
        nid = 1
        # x = view(X, i, :)
        @inbounds while tree.split[id]
            X[i, tree.feat[nid]] < tree.cond_float[id] ? nid <<= 1 : nid <<= 1 + 1
        end
        # @inbounds for k in 1:K
        pred[i,k] += tree.pred[1, nid]
        # end
    end
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu!(pred::AbstractMatrix{T}, tree::TreeGPU{T}, X::AbstractMatrix) where {T}
    K = size(pred, 2)
    thread_i = min(MAX_THREADS, length(𝑖))
    threads = thread_i
    blocks = length(𝑖) ÷ thread_i + 1
    @cuda blocks = blocks threads = threads predict_kernel!(pred, tree, X)

end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu(tree::TreeGPU{T}, X::AbstractMatrix, K) where {T}
    pred = CUDA.zeros(T, size(X, 1), K)
    predict_gpu!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{T,S}, X::AbstractMatrix) where {T,S}
    K = length(model.trees[1].nodes[1].pred)
    pred = CUDA.zeros(T, size(X, 1), K)
    for tree in model.trees
        predict_gpu!(pred, tree, X)
    end
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Gaussian
        pred[:,2] = exp.(pred[:,2])
    elseif typeof(model.params.loss) == Softmax
        pred = transpose(reshape(pred, model.K, :))
        for i in 1:size(pred, 1)
            pred[i,:] .= softmax(pred[i,:])
        end
    end
    return pred
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu(params::L, hist, j, nid) where {L <: GradientRegression,T}
    - params.η * hist[1,end,j,nid] / (hist[2,end,j,nid] + params.λ * hist[3,end,j,nid])
end

# prediction in Leaf - GaussianRegression
function pred_leaf_gpu(::L, node::TrainNodeGPU{T}, params::EvoTypes) where {L <: GaussianRegression,T}
    [- params.η * node.∑[1] / (node.∑[3] + params.λ * node.∑[5]), - params.η * node.∑[2] / (node.∑[4] + params.λ * node.∑[5])]
end
