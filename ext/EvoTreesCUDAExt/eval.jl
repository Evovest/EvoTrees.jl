using KernelAbstractions

########################
# MSE
########################
@kernel function eval_mse_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * (p[1, i] - y[i])^2
    end
    return nothing
end
function EvoTrees.mse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mse_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# RMSE
########################
EvoTrees.rmse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat} =
    sqrt(EvoTrees.mse(p, y, w, eval; MAX_THREADS, kwargs...))

########################
# MAE
########################
@kernel function eval_mae_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * abs(p[1, i] - y[i])
    end
    return nothing
end
function EvoTrees.mae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mae_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Logloss
########################
@kernel function eval_logloss_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = EvoTrees.sigmoid(p[1, i])
        @inbounds eval[i] = w[i] * (-y[i] * log(pred) + (y[i] - 1) * log(1 - pred))
    end
    return nothing
end
function EvoTrees.logloss(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_logloss_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Gaussian
########################
@kernel function eval_gaussian_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = -w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return nothing
end
function EvoTrees.gaussian_mle(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_gaussian_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Poisson Deviance
########################
@kernel function eval_poisson_kernel!(eval, p, y, w)
    i = @index(Global)
    ϵ = eps(eltype(p)(1e-7))
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i] * log(y[i] / pred + ϵ) + pred - y[i])
    end
    return nothing
end
function EvoTrees.poisson(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_poisson_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Gamma Deviance
########################
@kernel function eval_gamma_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (log(pred / y[i]) + y[i] / pred - 1)
    end
    return nothing
end
function EvoTrees.gamma(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_gamma_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Tweedie Deviance
########################
@kernel function eval_tweedie_kernel!(eval, p, y, w)
    i = @index(Global)
    rho = eltype(p)(1.5)
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * pred^(1 - rho) / (1 - rho) + pred^(2 - rho) / (2 - rho))
    end
    return nothing
end
function EvoTrees.tweedie(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_tweedie_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# mlogloss
########################
@kernel function eval_mlogloss_kernel!(eval, p, y, w)
    i = @index(Global)
    K = size(p, 1)
    if i <= length(y)
        isum = zero(eltype(p))
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return nothing
end
function EvoTrees.mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mlogloss_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

