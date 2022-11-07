"""
    predict_kernel!
"""
function predict_kernel!(
    ::Type{L},
    pred::AbstractMatrix{T},
    split,
    feat,
    cond_float,
    leaf_pred::AbstractMatrix{T},
    X::CuDeviceMatrix,
) where {L,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    K = size(pred, 1)
    @inbounds if idx <= size(pred, 2)
        @inbounds while split[nid]
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
function predict_kernel!(
    ::Type{L},
    pred::AbstractMatrix{T},
    split,
    feat,
    cond_float,
    leaf_pred::AbstractMatrix{T},
    X::CuDeviceMatrix,
) where {L<:GradientRegression,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        @inbounds while split[nid]
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
function predict_kernel!(
    ::Type{L},
    pred::AbstractMatrix{T},
    split,
    feat,
    cond_float,
    leaf_pred::AbstractMatrix{T},
    X::CuDeviceMatrix,
) where {L<:Logistic,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        @inbounds while split[nid]
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
function predict_kernel!(
    ::Type{L},
    pred::AbstractMatrix{T},
    split,
    feat,
    cond_float,
    leaf_pred::AbstractMatrix{T},
    X::CuDeviceMatrix,
) where {L<:MLE2P,T}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if idx <= size(pred, 2)
        @inbounds while split[nid]
            X[idx, feat[nid]] < cond_float[nid] ? nid = nid << 1 : nid = nid << 1 + 1
        end
        pred[1, idx] += leaf_pred[1, nid]
        pred[2, idx] = max(-15, pred[2, idx] + leaf_pred[2, nid])
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict!(
    pred::AbstractMatrix{T},
    tree::TreeGPU{L,K,T},
    X::AbstractMatrix;
    MAX_THREADS = 1024,
) where {L,K,T}
    n = size(pred, 2)
    threads = min(MAX_THREADS, n)
    blocks = ceil(Int, n / threads)
    @cuda blocks = blocks threads = threads predict_kernel!(
        L,
        pred,
        tree.split,
        tree.feat,
        tree.cond_float,
        tree.pred,
        X,
    )
    CUDA.synchronize()
end

# prediction from single tree - assign each observation to its final leaf
function predict(tree::TreeGPU{L,K,T}, X::AbstractMatrix) where {L,K,T}
    pred = CUDA.zeros(T, K, size(X, 1))
    predict!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(
    m::EvoTreeGPU{L,K,T},
    X::AbstractMatrix;
    ntree_limit = length(m.trees),
) where {L,K,T}
    pred = CUDA.zeros(T, K, size(X, 1))
    X_gpu = CuArray(X)
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    for i = 1:ntree_limit
        predict!(pred, m.trees[i], X_gpu)
    end
    if L == Logistic
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L == GaussianMLE
        pred[2, :] .= exp.(pred[2, :])
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end


# prediction in Leaf - GradientRegression
function pred_leaf_gpu!(
    p::AbstractMatrix{T},
    n,
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:GradientRegression,T}
    @allowscalar(p[1, n] = -params.eta * ∑[1] / (∑[2] + params.lambda * ∑[3]))
    return nothing
end

# prediction in Leaf - MLE2P
function pred_leaf_gpu!(
    p::AbstractMatrix{T},
    n,
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:MLE2P,T}
    @allowscalar(p[1, n] = -params.eta * ∑[1] / (∑[3] + params.lambda * ∑[5]))
    @allowscalar(p[2, n] = -params.eta * ∑[2] / (∑[4] + params.lambda * ∑[5]))
    return nothing
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_gpu!(
    p::AbstractMatrix{T},
    n,
    ∑::AbstractVector{T},
    params::EvoTypes{L,T},
) where {L<:MultiClassRegression,T}
    K = size(p, 1)
    @inbounds for k = 1:K
        @allowscalar(p[k, n] = -params.eta * ∑[k] / (∑[k+K] + params.lambda * ∑[end]))
    end
    return nothing
end
