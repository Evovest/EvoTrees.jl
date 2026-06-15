#####################
# MSE
#####################
function kernel_mse_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds ∇[1, i] = 2 * (p[i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MSE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_mse_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# MAE
#####################
function kernel_mae_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_mae_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Credibility
#####################
function kernel_cred_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        @inbounds ∇[2, i] = (y[i] - p[1, i])^2 * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{<:EvoTrees.Cred},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_cred_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Quantile
#####################
function kernel_quantile_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, alpha::T) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        diff = (y[i] - p[1, i])
        @inbounds ∇[1, i] = diff > 0 ? alpha * ∇[3, i] : (alpha - 1) * ∇[3, i]
        @inbounds ∇[2, i] = diff
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
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
# Logistic
#####################
function kernel_logloss_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = EvoTrees.sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_logloss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Poisson
#####################
function kernel_poisson_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Poisson},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_poisson_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Gamma
#####################
function kernel_gamma_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Gamma},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_gamma_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Tweedie
#####################
function kernel_tweedie_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Tweedie},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_tweedie_∇!(∇, p, y)
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
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(y)
        # first order gradients
        ∇[1, i] = (p[1, i] - y[i]) / exp(2 * p[2, i]) * ∇[5, i]
        ∇[2, i] = (1 - (p[1, i] - y[i])^2 / exp(2 * p[2, i])) * ∇[5, i]
        # # second order gradients
        ∇[3, i] = ∇[5, i] / exp(2 * p[2, i])
        ∇[4, i] = 2 * ∇[5, i] / exp(2 * p[2, i]) * (p[1, i] - y[i])^2
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.GaussianMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_gauss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_gauss_mt_∇!(∇, p, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            inv = 1 / exp(2 * ls)
            d = μ - yt
            ∇[gb+1, i] = d * inv * w
            ∇[gb+2, i] = (1 - d^2 * inv) * w
            ∇[hb+1, i] = inv * w
            ∇[hb+2, i] = 2 * inv * d^2 * w
        end
    end
    return
end

function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.GaussianMLE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks=blocks threads=threads kernel_gauss_mt_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_logistic_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(y)
        w = ∇[5, i]
        μ = p[1, i]
        ls = p[2, i]
        yt = y[i]
        ∇[1, i] = -tanh((yt - μ) / (2 * exp(ls))) * exp(-ls) * w
        ∇[2, i] = -(exp(-ls) * (yt - μ) * tanh((yt - μ) / (2 * exp(ls))) - 1) * w
        ∇[3, i] = sech((yt - μ) / (2 * exp(ls)))^2 / (2 * exp(2 * ls)) * w
        ∇[4, i] = (exp(-2 * ls) * (μ - yt) *
                   (μ - yt + exp(ls) * sinh(exp(-ls) * (μ - yt)))) /
                  (1 + cosh(exp(-ls) * (μ - yt))) * w
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogisticMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_logistic_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_logistic_mt_∇!(∇, p, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            ∇[gb+1, i] = -tanh((yt - μ) / (2 * exp(ls))) * exp(-ls) * w
            ∇[gb+2, i] = -(exp(-ls) * (yt - μ) * tanh((yt - μ) / (2 * exp(ls))) - 1) * w
            ∇[hb+1, i] = sech((yt - μ) / (2 * exp(ls)))^2 / (2 * exp(2 * ls)) * w
            ∇[hb+2, i] = (exp(-2 * ls) * (μ - yt) *
                          (μ - yt + exp(ls) * sinh(exp(-ls) * (μ - yt)))) /
                         (1 + cosh(exp(-ls) * (μ - yt))) * w
        end
    end
    return
end

function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.LogisticMLE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks=blocks threads=threads kernel_logistic_mt_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_gradreg_∇!(∇, p, y, ::Type{EvoTrees.MSE})
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            ∇[k, i]   = 2 * (p[k, i] - y[k, i]) * w
            ∇[K+k, i] = 2 * w
        end
    end
    return
end
function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.MSE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks = blocks threads = threads kernel_gradreg_∇!(∇, p, y, EvoTrees.MSE)
    CUDA.synchronize()
    return
end
