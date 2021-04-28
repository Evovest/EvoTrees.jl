function predict_kernel!(pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds for k in 1:K
            pred[k, idx] += leaf_pred[k, nid]
        end
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict!(loss::L, pred::AbstractMatrix{T}, tree::TreeGPU{T}, X::AbstractMatrix, K; MAX_THREADS=512) where {L,T}
    K = size(pred, 1)
    n = size(pred, 2)
    thread_i = min(MAX_THREADS, n)
    threads = thread_i
    blocks = ceil(Int, n / thread_i)
    @cuda blocks = blocks threads = threads predict_kernel!(pred, tree.split, tree.feat, tree.cond_float, tree.pred, X, K)
    CUDA.synchronize()
end

# prediction from single tree - assign each observation to its final leaf
function predict(loss::L, tree::TreeGPU{T}, X::AbstractMatrix, K) where {L,T}
    pred = CUDA.zeros(T, K, size(X, 1))
    predict!(loss, pred, tree, X, K)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{T,S}, X::AbstractMatrix) where {T,S}
    K = size(model.trees[1].pred, 1)
    pred = CUDA.zeros(T, K, size(X, 1))
    X_gpu = CuArray(X)
    for tree in model.trees
        predict!(model.params.loss, pred, tree, X_gpu, model.K)
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
    return nothing
end

# prediction in Leaf - Gaussian
function pred_leaf_gpu!(::S, pred::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes) where {S <: GaussianRegression,T}
    pred[1,n] = - params.η * ∑[1] / (∑[3] + params.λ * ∑[5])
    pred[2,n] = - params.η * ∑[2] / (∑[4] + params.λ * ∑[5])
    return nothing
end
