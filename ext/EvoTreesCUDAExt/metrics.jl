@inline _metric_value(::Type{EvoTrees.MSE}, pk, yk, alpha) = (pk - yk)^2
@inline _metric_value(::Type{EvoTrees.MAE}, pk, yk, alpha) = abs(pk - yk)

@inline function _metric_value(::Type{EvoTrees.Quantile}, pk, yk, alpha)
    return alpha * max(yk - pk, zero(pk)) + (1 - alpha) * max(pk - yk, zero(pk))
end

@inline function _metric_value(::Type{EvoTrees.LogLoss}, pk, yk, alpha)
    pred = EvoTrees.sigmoid(pk)
    return -yk * log(pred) + (yk - 1) * log(1 - pred)
end

@inline function _metric_value(::Type{EvoTrees.Poisson}, pk, yk, alpha)
    pred = exp(pk)
    ϵ = eps(typeof(pk)(1e-7))
    return 2 * (yk * log(yk / pred + ϵ) + pred - yk)
end

@inline function _metric_value(::Type{EvoTrees.Gamma}, pk, yk, alpha)
    pred = exp(pk)
    return 2 * (log(pred / yk) + yk / pred - 1)
end

@inline function _metric_value(::Type{EvoTrees.Tweedie}, pk, yk, alpha)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return 2 * (yk^(2 - rho) / (1 - rho) / (2 - rho) - yk * pred^(1 - rho) / (1 - rho) + pred^(2 - rho) / (2 - rho))
end

@inline _mle2p_metric_value(::Type{EvoTrees.GaussianMLE}, μ, ls, yt) = -(ls + (yt - μ)^2 / (2 * exp(2 * ls)))
@inline _mle2p_metric_value(::Type{EvoTrees.LogisticMLE}, μ, ls, yt) = log(1 / 4 * sech(exp(-ls) * (yt - μ) / 2)^2) - ls

########################
# Pointwise metrics
########################
@inline _metric_target_count(p, y::CuDeviceVector) = 1
@inline _metric_target_count(p, y::CuDeviceMatrix) = size(p, 1)

function eval_metric_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y, w::CuDeviceVector{T}, ::Type{M}, alpha::T) where {T<:AbstractFloat,M}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(w)
        K = _metric_target_count(p, y)
        acc = zero(T)
        @inbounds for k in 1:K
            acc += _metric_value(M, p[k, i], _cuda_target(y, k, i), alpha)
        end
        @inbounds eval[i] = w[i] * acc / K
    end
    return nothing
end

function _eval_metric(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}, ::Type{M}; MAX_THREADS=1024, alpha=0.5, kwargs...) where {T<:AbstractFloat,M}
    threads = min(MAX_THREADS, length(w))
    blocks = cld(length(w), threads)
    @cuda blocks = blocks threads = threads eval_metric_kernel!(eval, p, y, w, M, T(alpha))
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

EvoTrees.mse(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.MSE; kwargs...)

EvoTrees.rmse(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    sqrt(EvoTrees.mse(p, y, w, eval; kwargs...))

EvoTrees.mae(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.MAE; kwargs...)

EvoTrees.wmae(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.Quantile; kwargs...)

########################
# MultiQuantile
########################
function eval_multiquantile_kernel!(
    eval::CuDeviceVector{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceVector{T},
    w::CuDeviceVector{T},
    alphas::CuDeviceVector{T},
    K::Int,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        yi = y[i]
        acc = zero(T)
        @inbounds for k in 1:K
            alpha = alphas[k]
            acc += _metric_value(EvoTrees.Quantile, p[k, i], yi, alpha)
        end
        @inbounds eval[i] = w[i] * acc / K
    end
    return nothing
end

function EvoTrees.multiquantile(
    p::CuMatrix{T},
    y::CuVector{T},
    w::CuVector{T},
    eval::CuVector{T};
    MAX_THREADS=1024,
    alphas,
    kwargs...
) where {T<:AbstractFloat}
    K = length(alphas)
    alphas_dev = alphas isa CuVector ? alphas : CuArray(T.(alphas))
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_multiquantile_kernel!(eval, p, y, w, alphas_dev, K)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

EvoTrees.logloss(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.LogLoss; kwargs...)

EvoTrees.poisson(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.Poisson; kwargs...)

EvoTrees.gamma(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.Gamma; kwargs...)

EvoTrees.tweedie(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_metric(p, y, w, eval, EvoTrees.Tweedie; kwargs...)

########################
# Two-parameter MLE metrics
########################
@inline _mle2p_target_count(p, y::CuDeviceVector) = 1
@inline _mle2p_target_count(p, y::CuDeviceMatrix) = size(p, 1) ÷ 2

function eval_mle2p_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y, w::CuDeviceVector{T}, ::Type{M}) where {T<:AbstractFloat,M<:EvoTrees.MLE2P}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(w)
        Y = _mle2p_target_count(p, y)
        acc = zero(T)
        @inbounds for t in 1:Y
            acc += _mle2p_metric_value(M, p[2t-1, i], p[2t, i], _cuda_target(y, t, i))
        end
        @inbounds eval[i] = w[i] * acc / Y
    end
    return nothing
end

function _eval_mle2p_metric(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}, ::Type{M}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat,M<:EvoTrees.MLE2P}
    threads = min(MAX_THREADS, length(w))
    blocks = cld(length(w), threads)
    @cuda blocks = blocks threads = threads eval_mle2p_kernel!(eval, p, y, w, M)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

EvoTrees.gaussian_mle(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_mle2p_metric(p, y, w, eval, EvoTrees.GaussianMLE; kwargs...)

EvoTrees.logistic_mle(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_mle2p_metric(p, y, w, eval, EvoTrees.LogisticMLE; kwargs...)

########################
# mlogloss
########################
function eval_mlogloss_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector, w::CuDeviceVector{T}) where {T<:AbstractFloat}
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

function EvoTrees.mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_mlogloss_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

