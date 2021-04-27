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

function predict_kernel!(pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix) where {T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        # @inbounds for k in 1:K
        pred[1, idx] += leaf_pred[1, nid]
        # end
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu!(pred::AbstractMatrix{T}, tree, X_bin::AbstractMatrix; MAX_THREADS=512) where {T}
    K = size(pred, 1)
    n = size(pred, 2)
    thread_i = min(MAX_THREADS, n)
    threads = thread_i
    blocks = ceil(Int, n / thread_i)
    @cuda blocks = blocks threads = threads predict_kernel!(pred, tree.split, tree.feat, tree.cond_float, tree.pred, X_bin)
    CUDA.synchronize()
end

# prediction from single tree - assign each observation to its final leaf
function predict_gpu(tree::TreeGPU{T}, X::AbstractMatrix, K) where {T}
    pred = CUDA.zeros(T, K, size(X, 1))
    predict_gpu!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{T,S}, X::AbstractMatrix) where {T,S}
    K = size(model.trees[1].pred, 1)
    pred = CUDA.zeros(T, K, size(X, 1))
    X_gpu = CuArray(X)
    for tree in model.trees
        predict_gpu!(pred, tree, X_gpu)
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
    return Array(pred')
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu!(::S, pred::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes) where {S <: GradientRegression,T}
    pred[1,n] = - params.η * ∑[1] / (∑[2] + params.λ * ∑[3])
end
