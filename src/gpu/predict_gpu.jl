"""
    predict_kernel!
"""
function predict_kernel!(::Type{L}, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L,T}
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
function predict_kernel!(::Type{L}, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:GradientRegression,T}
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
function predict_kernel!(::Type{L}, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:Logistic,T}
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
        MLE2P
"""
function predict_kernel!(::Type{L}, pred::AbstractMatrix{T}, split, feat, cond_float, leaf_pred::AbstractMatrix{T}, X::CuDeviceMatrix, K) where {L<:MLE2P,T}
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
function predict!(pred::AbstractMatrix{T}, tree::TreeGPU{L,T}, X::AbstractMatrix, K; MAX_THREADS=1024) where {L,T}
    K = size(pred, 1)
    n = size(pred, 2)
    threads = min(MAX_THREADS, n)
    blocks = ceil(Int, n / threads)
    @cuda blocks = blocks threads = threads predict_kernel!(L, pred, tree.split, tree.feat, tree.cond_float, tree.pred, X, K)
    CUDA.synchronize()
end

# prediction from single tree - assign each observation to its final leaf
function predict(tree::TreeGPU{L,T}, X::AbstractMatrix, K) where {L,T}
    pred = CUDA.zeros(T, K, size(X, 1))
    predict!(pred, tree, X, K)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTreeGPU{L,T,S}, X::AbstractMatrix) where {L,T,S}
    K = size(model.trees[1].pred, 1)
    pred = CUDA.zeros(T, K, size(X, 1))
    X_gpu = CuArray(X)
    for tree in model.trees
        predict!(pred, tree, X_gpu, model.K)
    end
    if L == Logistic
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L == GaussianDist
        pred[2, :] .= exp.(pred[2, :])
    elseif L == Softmax
        pred = transpose(reshape(pred, model.K, :))
        for i in axes(pred, 1)
            pred[i, :] .= softmax(pred[i, :])
        end
    end
    pred = model.K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu!(p::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes{L,T,S}) where {L<:GradientRegression,T,S}
    @allowscalar(p[1, n] = -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3]))
    return nothing
end

# prediction in Leaf - MLE2P
function pred_leaf_gpu!(p::AbstractMatrix{T}, n, ∑::AbstractVector{T}, params::EvoTypes{L,T,S}) where {L<:MLE2P,T,S}
    @allowscalar(p[1, n] = -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5]))
    @allowscalar(p[2, n] = -params.eta * ∑[2] / (∑[4] + params.lambda * ∑[5]))
    return nothing
end