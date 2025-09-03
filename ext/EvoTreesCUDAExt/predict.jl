@kernel function predict_kernel!(
    ::Type{L},
    pred,
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
) where {L}
    i = @index(Global)
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
    return nothing
end

# GradientRegression - K=1 fast path
@kernel function predict_kernel!(
    ::Type{<:EvoTrees.GradientRegression},
    pred,
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
)
    i = @index(Global)
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        pred[1, i] += leaf_pred[1, nid]
    end
    return nothing
end

# Logistic - clamp linear predictor per update
@kernel function predict_kernel!(
    ::Type{<:EvoTrees.LogLoss},
    pred,
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
)
    i = @index(Global)
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        val = pred[1, i] + leaf_pred[1, nid]
        pred[1, i] = min(eltype(pred)(15), max(eltype(pred)(-15), val))
    end
    return nothing
end

# MLE2P - clamp second parameter per update
@kernel function predict_kernel!(
    ::Type{<:EvoTrees.MLE2P},
    pred,
    split,
    feats,
    cond_bins,
    leaf_pred,
    x_bin,
    feattypes,
)
    i = @index(Global)
    nid = 1
    @inbounds if i <= size(pred, 2)
        @inbounds while split[nid]
            feat = feats[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= cond_bins[nid] : x_bin[i, feat] == cond_bins[nid]
            nid = (nid << 1) + Int(!cond)
        end
        pred[1, i] += leaf_pred[1, nid]
        val2 = pred[2, i] + leaf_pred[2, nid]
        pred[2, i] = max(eltype(pred)(-15), val2)
    end
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
    backend = KernelAbstractions.get_backend(pred)

    split_dev = KernelAbstractions.zeros(backend, eltype(tree.split), length(tree.split))
    feats_dev = KernelAbstractions.zeros(backend, eltype(tree.feat), length(tree.feat))
    cond_dev = KernelAbstractions.zeros(backend, eltype(tree.cond_bin), length(tree.cond_bin))
    leaf_dev = KernelAbstractions.zeros(backend, eltype(tree.pred), size(tree.pred,1), size(tree.pred,2))
    copyto!(split_dev, tree.split)
    copyto!(feats_dev, tree.feat)
    copyto!(cond_dev, tree.cond_bin)
    copyto!(leaf_dev, tree.pred)

    workgroupsize = min(256, n)
    predict_kernel!(backend)(L, pred, split_dev, feats_dev, cond_dev, leaf_dev, x_bin, feattypes; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
end

function EvoTrees.predict!(
    pred::CuMatrix{T},
    tree::EvoTrees.Tree{L,K},
    x_bin::CuMatrix,
    feattypes::CuVector{Bool};
    MAX_THREADS=1024
) where {L<:EvoTrees.MLogLoss,K,T}
    n = size(pred, 2)
    backend = KernelAbstractions.get_backend(pred)

    split_dev = KernelAbstractions.zeros(backend, eltype(tree.split), length(tree.split))
    feats_dev = KernelAbstractions.zeros(backend, eltype(tree.feat), length(tree.feat))
    cond_dev = KernelAbstractions.zeros(backend, eltype(tree.cond_bin), length(tree.cond_bin))
    leaf_dev = KernelAbstractions.zeros(backend, eltype(tree.pred), size(tree.pred,1), size(tree.pred,2))
    copyto!(split_dev, tree.split)
    copyto!(feats_dev, tree.feat)
    copyto!(cond_dev, tree.cond_bin)
    copyto!(leaf_dev, tree.pred)

    workgroupsize = min(256, n)
    predict_kernel!(backend)(L, pred, split_dev, feats_dev, cond_dev, leaf_dev, x_bin, feattypes; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)

    pred .= max.(T(-15), pred .- maximum(pred, dims=1))
end

# prediction for EvoTree model
function EvoTrees._predict(
    m::EvoTrees.EvoTree,
    data,
    ::Type{<:EvoTrees.GPU};
    ntree_limit=length(m.trees))

    K = m.K
    xb = EvoTrees.binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    backend = KernelAbstractions.get_backend(CuArray(xb))
    x_bin = KernelAbstractions.zeros(backend, size(xb,1), size(xb,2))
    copyto!(x_bin, CuArray(xb))
    ft = m.info[:feattypes]
    feattypes = KernelAbstractions.zeros(backend, Bool, length(ft))
    copyto!(feattypes, ft)

    pred = KernelAbstractions.zeros(backend, K, size(data, 1))
    ntrees = length(m.trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")

    for i = 1:ntree_limit
        EvoTrees.predict!(pred, m.trees[i], x_bin, feattypes)
    end
    
    # Apply loss-specific transformations
    L = m.loss_type
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

@kernel function softmax_kernel!(p)
    i = @index(Global)
    K, nobs = size(p)
    if i <= nobs
        isum = zero(eltype(p))
        @inbounds for k in 1:K
            p[k, i] = exp(p[k, i])
            isum += p[k, i]
        end
        @inbounds for k in 1:K
            p[k, i] /= isum
        end
    end
    return nothing
end

function EvoTrees.softmax!(p::CuMatrix{T}; MAX_THREADS=1024) where {T}
    K, nobs = size(p)
    backend = KernelAbstractions.get_backend(p)
    workgroupsize = min(256, nobs)
    softmax_kernel!(backend)(p; ndrange=nobs, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return nothing
end

