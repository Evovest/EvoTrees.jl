"""
    predict_kernel!
"""
function predict_kernel!(
    ::Type{L},
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {L,T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = nid << 1 + !cond
        end
        @inbounds for k = 1:K
            pred[k, i] += leaf_pred[k, nid]
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
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {L<:GradientRegression,T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = nid << 1 + !cond
        end
        pred[1, i] += leaf_pred[1, nid]
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
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {L<:Logistic,T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = nid << 1 + !cond
        end
        pred[1, i] = min(T(15), max(T(-15), pred[1, i] + leaf_pred[1, nid]))
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
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {L<:MLE2P,T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = nid << 1 + !cond
        end
        pred[1, i] += leaf_pred[1, nid]
        pred[2, i] = max(T(-15), pred[2, i] + leaf_pred[2, nid])
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function predict!(
    pred::CuMatrix{T},
    tree::Tree{L,K,T},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
    MAX_THREADS=1024
) where {L,K,T}
    n = size(pred, 2)
    threads = min(MAX_THREADS, n)
    blocks = cld(n, threads)
    @cuda blocks = blocks threads = threads predict_kernel!(
        L,
        pred,
        CuArray(tree.split),
        CuArray(tree.feat),
        CuArray(tree.cond_bin),
        CuArray(tree.pred),
        x_bin,
        feattypes,
    )
    CUDA.synchronize()
end
function predict!(
    pred::AbstractMatrix{T},
    tree::Tree{L,K,T},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
    MAX_THREADS=1024
) where {L<:Softmax,K,T}
    n = size(pred, 2)
    threads = min(MAX_THREADS, n)
    blocks = cld(n, threads)
    @cuda blocks = blocks threads = threads predict_kernel!(
        L,
        pred,
        CuArray(tree.split),
        CuArray(tree.feat),
        CuArray(tree.cond_bin),
        CuArray(tree.pred),
        x_bin,
        feattypes,
    )
    CUDA.synchronize()
    pred .= max.(T(-15), pred .- maximum(pred, dims=1))
end

# prediction from single tree - assign each observation to its final leaf
function predict(
    m::EvoTreeGPU{L,K,T},
    data::Union{AbstractMatrix, AbstractDataFrame};
    ntree_limit=length(m.trees)
) where {L,K,T}
    pred = CUDA.zeros(T, K, size(data, 1))
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    x_bin = CuArray(binarize(data; fnames=m.info[:fnames], edges=m.info[:edges]))
    feattypes = CuArray(m.info[:feattypes])
    for i = 1:ntree_limit
        predict!(pred, m.trees[i], x_bin, feattypes)
    end
    if L == Logistic
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L in [GaussianMLE, LogisticMLE]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == Softmax
        # @inbounds for i in axes(pred, 2)
        #     pred[:, i] .= softmax(pred[:, i])
        # end
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end
