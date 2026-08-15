@inline _cuda_target(y::CuDeviceVector, k, i) = y[i]
@inline _cuda_target(y::CuDeviceMatrix, k, i) = y[k, i]

#####################
# MAE
#####################
function kernel_mae_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            ∇[k, i] = (_cuda_target(y, k, i) - p[k, i]) * w
            ∇[K+k, i] = zero(T)
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::Union{CuVector,CuMatrix},
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_mae_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Credibility
#####################
function kernel_cred_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            d = _cuda_target(y, k, i) - p[k, i]
            ∇[k, i] = d * w
            ∇[K+k, i] = d^2 * w
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::Union{CuVector,CuMatrix},
    ::Type{<:EvoTrees.Cred},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_cred_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Quantile
#####################
function kernel_quantile_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y, alpha::T) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            diff = _cuda_target(y, k, i) - p[k, i]
            ∇[k, i] = diff > 0 ? alpha * w : (alpha - 1) * w
            ∇[K+k, i] = diff
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::Union{CuVector{T},CuMatrix{T}},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_quantile_∇!(∇, p, y, T(params.alpha))
    CUDA.synchronize()
    return
end

#####################
# MultiQuantile
#####################
function kernel_multiquantile_∇!(
    ∇::CuDeviceMatrix{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceVector{T},
    alphas::CuDeviceVector{T},
    K::Int,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        yi = y[i]
        w_idx = 2 * K + 1
        @inbounds wi = ∇[w_idx, i]
        @inbounds for k in 1:K
            diff = yi - p[k, i]
            alpha = alphas[k]
            ∇[k, i] = diff > 0 ? alpha * wi : (alpha - 1) * wi
            ∇[K + k, i] = diff
        end
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.MultiQuantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {T<:AbstractFloat}
    K = length(params.alphas)
    alphas = CuArray(T.(params.alphas))
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_multiquantile_∇!(∇, p, y, alphas, K)
    CUDA.synchronize()
    return
end

#####################
# Softmax
#####################
function kernel_mlogloss_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K = size(p, 1)
    if i <= length(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds for k in 1:K
            iexp = exp(p[k, i])
            if k == y[i]
                ∇[k, i] = (iexp / isum - 1) * ∇[end, i]
            else
                ∇[k, i] = iexp / isum * ∇[end, i]
            end
            ∇[k+K, i] = 1 / isum * (1 - iexp / isum) * ∇[end, i]
        end
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MLogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_mlogloss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

################################################################################
# Two-parameter MLE: Fisher information, scale = softplus(φ)
################################################################################
function kernel_mle2p_∇!(∇, p, y, ::Type{L}) where {L<:EvoTrees.MLE2P}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            g1, g2, h1, h2 = EvoTrees.mle2p_grad_hess(L, p[2t-1, i], p[2t, i], _cuda_target(y, t, i))
            ∇[gb+1, i] = g1 * w
            ∇[gb+2, i] = g2 * w
            ∇[hb+1, i] = h1 * w
            ∇[hb+2, i] = h2 * w
        end
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::Union{CuVector,CuMatrix},
    ::Type{L},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {L<:EvoTrees.MLE2P}
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_mle2p_∇!(∇, p, y, L)
    CUDA.synchronize()
    return
end

@inline gradreg_grad_hess(::Type{EvoTrees.MSE}, pk, yk) = (2 * (pk - yk), 2 * one(pk))
@inline function gradreg_grad_hess(::Type{EvoTrees.LogLoss}, pk, yk)
    pred = EvoTrees.sigmoid(pk)
    return (pred - yk, pred * (1 - pred))
end
@inline gradreg_grad_hess(::Type{EvoTrees.Poisson}, pk, yk) = (exp(pk) - yk, exp(pk))
@inline function gradreg_grad_hess(::Type{EvoTrees.Gamma}, pk, yk)
    pred = exp(pk)
    return (2 * (1 - yk / pred), 2 * yk / pred)
end
@inline function gradreg_grad_hess(::Type{EvoTrees.Tweedie}, pk, yk)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return (
        2 * (pred^(2 - rho) - yk * pred^(1 - rho)),
        2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * yk * pred^(1 - rho)),
    )
end

function kernel_gradreg_∇!(∇, p, y, ::Type{L}) where {L<:EvoTrees.GradientRegression}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, h = gradreg_grad_hess(L, p[k, i], _cuda_target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
    return
end
function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix}, ::Type{L}, params::EvoTrees.EvoTypes; MAX_THREADS=1024) where {L<:EvoTrees.GradientRegression}
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_gradreg_∇!(∇, p, y, L)
    CUDA.synchronize()
    return
end
