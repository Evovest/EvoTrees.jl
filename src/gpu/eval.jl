"""
    MSE
"""
function eval_mse_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = w[i] * (p[1, i] - y[i])^2
    end
    return nothing
end
function mse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_mse_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    RMSE
"""
rmse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat} =
    sqrt(rmse(p, y, w; MAX_THREADS, kwargs...))

"""
    MAE
"""
function eval_mae_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = w[i] * abs(p[1, i] - y[i])
    end
    return nothing
end
function mae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_mae_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    Logloss
"""
function eval_logloss_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds eval[i] = w[i] * (-y[i] * log(pred) + (y[i] - 1) * log(1 - pred))
    end
    return nothing
end
function logloss(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_logloss_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end


"""
    Gaussian
"""
function eval_gaussian_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = -w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return nothing
end
function gaussian_mle(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_gaussian_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    Poisson Deviance
"""
function eval_poisson_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ϵ = eps(T(1e-7))
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i] * log(y[i] / pred + ϵ) + pred - y[i])
    end
    return nothing
end

function poisson(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_poisson_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    Gamma Deviance
"""
function eval_gamma_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (log(pred / y[i]) + y[i] / pred - 1)
    end
    return nothing
end

function gamma(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_gamma_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    Tweedie Deviance
"""
function eval_tweedie_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    rho = T(1.5)
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * pred^(1 - rho) / (1 - rho) + pred^(2 - rho) / (2 - rho))

    end
    return nothing
end

function tweedie(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_tweedie_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end


"""
    mlogloss
"""
function mlogloss_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K = size(p, 1)
    if i <= length(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return nothing
end

function mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads mlogloss_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end