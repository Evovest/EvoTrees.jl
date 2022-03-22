"""
    predict_kernel!
"""
function predict_kernel!(::L, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        @inbounds for k = 1:K
            pred[k, idx] += leaf_pred[k, nid]
        end
    end
    sync_threads()
    return nothing
end

"""
    predict_kernel!
        GradientRegression
"""
function predict_kernel!(::L, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:GradientRegression,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        pred[1, idx] += leaf_pred[1, nid]
    end
    sync_threads()
    return nothing
end

"""
    predict_kernel!
        Logistic
"""
function predict_kernel!(::L, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:Logistic,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        pred[1, idx] = min(15, max(-15, pred[1, idx] + leaf_pred[1, nid]))
    end
    sync_threads()
    return nothing
end

"""
    predict_kernel!
        GaussianRegression
"""
function predict_kernel!(::L, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:GaussianRegression,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        pred[1, idx] += leaf_pred[1, nid]
        pred[2, idx] = max(-15, pred[2, idx] + leaf_pred[2, nid])
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict!(loss::L, pred::AbstractMatrix{T}, tree::TreeGPU{T}, X::AbstractMatrix, K; MAX_THREADS = 1024) where {L,T}
    K = size(pred, 1)
    n = size(pred, 2)
    threads = min(MAX_THREADS, n)
    blocks = ceil(Int, n / threads)
    @cuda blocks = blocks threads = threads predict_kernel!(loss, pred, tree.split, tree.feat, tree.cond_float, tree.pred, X, K)
    CUDA.synchronize()
end

# prediction from single tree - assign each observation to its final leaf
function predict(loss::L, tree::TreeGPU{T}, X::AbstractMatrix, K) where {L,T}
    pred = CUDA.zeros(T, K, size(X, 1))
    predict!(loss, pred, tree, X, K)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{T}, X::AbstractMatrix) where {T}
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
        pred[2, :] = exp.(pred[2, :])
    elseif typeof(model.params.loss) == Softmax
        pred = transpose(reshape(pred, model.K, :))
        for i = 1:size(pred, 1)
            pred[i, :] .= softmax(pred[i, :])
        end
    end
    return Array(pred')
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu!(::S, p::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes) where {S<:GradientRegression,T}
    @allowscalar(p[1, n] = -params.η * ∑[1] / (∑[2] + params.λ * ∑[3]))
    return nothing
end

# prediction in Leaf - Gaussian
function pred_leaf_gpu!(::S, p::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes) where {S<:GaussianRegression,T}
    p[1, n] = -params.η * ∑[1] / (∑[3] + params.λ * ∑[5])
    p[2, n] = -params.η * ∑[2] / (∑[4] + params.λ * ∑[5])
    return nothing
end
