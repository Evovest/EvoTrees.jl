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
    K = size(pred, 1)
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        @inbounds for k = 1:K
            pred[k, i] += leaf_pred[k, nid]
        end
    end
    sync_threads()
    return nothing
end

# GradientRegression
function predict_kernel!(
    ::Type{<:EvoTrees.GradientRegression},
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        pred[1, i] += leaf_pred[1, nid]
    end
    sync_threads()
    return nothing
end

# Logistic
function predict_kernel!(
    ::Type{<:EvoTrees.LogLoss},
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        pred[1, i] += leaf_pred[1, nid]
    end
    sync_threads()
    return nothing
end

# MLE2P
function predict_kernel!(
    ::Type{<:EvoTrees.MLE2P},
    pred::CuDeviceMatrix{T},
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        pred[1, i] += leaf_pred[1, nid]
        pred[2, i] = max(T(-15), pred[2, i] + leaf_pred[2, nid])
    end
    sync_threads()
    return nothing
end

# prediction from single tree - assign each observation to its final leaf
function EvoTrees.predict!(
    pred::CuMatrix{T},
    tree::EvoTrees.Tree{L,K},
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

function EvoTrees.predict!(
    pred::CuMatrix{T},
    tree::EvoTrees.Tree{L,K},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
    MAX_THREADS=1024
) where {L<:EvoTrees.MLogLoss,K,T}
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

# prediction for EvoTree model
function EvoTrees._predict(
    m::EvoTrees.EvoTree{L,K},
    data,
    ::Type{<:EvoTrees.GPU};
    ntree_limit=length(m.trees)) where {L,K}

    EvoTrees.Tables.istable(data) ? data = EvoTrees.Tables.columntable(data) : nothing
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    x_bin = CuArray(EvoTrees.binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges]))
    nobs = size(x_bin, 1)
    pred = CUDA.zeros(K, nobs)
    feattypes = CuArray(m.info[:feattypes])
    for i = 1:ntree_limit
        EvoTrees.predict!(pred, m.trees[i], x_bin, feattypes)
    end
    if L == EvoTrees.LogLoss
        pred .= EvoTrees.sigmoid.(pred)
    elseif L ∈ [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        pred .= exp.(pred)
    elseif L in [EvoTrees.GaussianMLE, EvoTrees.LogisticMLE]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == EvoTrees.MLogLoss
        EvoTrees.softmax!(pred)
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end

function softmax_kernel!(p::CuDeviceMatrix{T}) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K, nobs = size(p)
    if i <= nobs
        isum = zero(T)
        @inbounds for k in 1:K
            p[k, i] = exp(p[k, i])
            isum += exp(p[k, i])
        end
        @inbounds for k in 1:K
            p[k, i] /= isum
        end
    end
    return nothing
end

function EvoTrees.softmax!(p::CuMatrix{T}; MAX_THREADS=1024) where {T}
    K, nobs = size(p)
    threads = min(MAX_THREADS, nobs)
    blocks = cld(nobs, threads)
    @cuda blocks = blocks threads = threads softmax_kernel!(p)
    CUDA.synchronize()
    return nothing
end

# Quantile - special case where ∇ is passed as argument
function quantile_gpu(x::AnyCuVector, alpha)
    x_sort = sort(x)
    idx = ceil(Int, alpha * length(x_sort))
    return CUDA.@allowscalar x_sort[idx]
end

function EvoTrees.pred_leaf_cpu!(p::Matrix, n, ∑::AbstractVector{T}, ::Type{L}, params::EvoTrees.EvoTypes, ∇::CuMatrix, is) where {L<:EvoTrees.Quantile,T}
    ϵ = eps(T)
    p[1, n] = params.eta / params.bagging_size * quantile_gpu(view(∇, 2, is), params.alpha) / (1 + params.lambda + params.L2 / ∑[3])
end
