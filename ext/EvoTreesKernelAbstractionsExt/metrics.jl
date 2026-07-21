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
@inline _mle2p_metric_value(::Type{EvoTrees.LogisticMLE}, μ, ls, yt) = log(1 / 4 * sech(exp(-ls) * (yt - μ))^2) - ls

########################
# Pointwise metrics
########################
@inline _metric_target_count(p, y::AbstractVector) = 1
@inline _metric_target_count(p, y::AbstractMatrix) = size(p, 1)

@kernel function eval_metric_kernel!(eval, @Const(p), @Const(y), @Const(w), ::Type{M}, alpha) where {M}
    i = @index(Global, Linear)
    @inbounds if i <= length(w)
        K = _metric_target_count(p, y)
        acc = zero(eltype(eval))
        for k in 1:K
            acc += _metric_value(M, p[k, i], _dev_target(y, k, i), alpha)
        end
        eval[i] = w[i] * acc / K
    end
end

function _eval_metric(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}, ::Type{M}; alpha=0.5, kwargs...) where {T<:AbstractFloat,M}
    backend = get_backend(eval)
    eval_metric_kernel!(backend)(eval, p, y, w, M, T(alpha); ndrange=length(w))
    KernelAbstractions.synchronize(backend)
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
@kernel function eval_multiquantile_kernel!(eval, @Const(p), @Const(y), @Const(w), @Const(alphas), K::Int)
    i = @index(Global, Linear)
    @inbounds if i <= length(y)
        yi = y[i]
        acc = zero(eltype(eval))
        for k in 1:K
            acc += _metric_value(EvoTrees.Quantile, p[k, i], yi, alphas[k])
        end
        eval[i] = w[i] * acc / K
    end
end

function EvoTrees.multiquantile(
    p::CuMatrix{T},
    y::CuVector{T},
    w::CuVector{T},
    eval::CuVector{T};
    alphas,
    kwargs...
) where {T<:AbstractFloat}
    K = length(alphas)
    backend = get_backend(eval)
    alphas_dev = alphas isa CuVector ? alphas : _to_device(backend, T.(alphas))
    eval_multiquantile_kernel!(backend)(eval, p, y, w, alphas_dev, K; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
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
@inline _mle2p_target_count(p, y::AbstractVector) = 1
@inline _mle2p_target_count(p, y::AbstractMatrix) = size(p, 1) ÷ 2

@kernel function eval_mle2p_kernel!(eval, @Const(p), @Const(y), @Const(w), ::Type{M}) where {M<:EvoTrees.MLE2P}
    i = @index(Global, Linear)
    @inbounds if i <= length(w)
        Y = _mle2p_target_count(p, y)
        acc = zero(eltype(eval))
        for t in 1:Y
            acc += _mle2p_metric_value(M, p[2t-1, i], p[2t, i], _dev_target(y, t, i))
        end
        eval[i] = w[i] * acc / Y
    end
end

function _eval_mle2p_metric(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}, ::Type{M}; kwargs...) where {T<:AbstractFloat,M<:EvoTrees.MLE2P}
    backend = get_backend(eval)
    eval_mle2p_kernel!(backend)(eval, p, y, w, M; ndrange=length(w))
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

EvoTrees.gaussian_mle(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_mle2p_metric(p, y, w, eval, EvoTrees.GaussianMLE; kwargs...)

EvoTrees.logistic_mle(p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}}, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat} =
    _eval_mle2p_metric(p, y, w, eval, EvoTrees.LogisticMLE; kwargs...)

########################
# mlogloss
########################
@kernel function eval_mlogloss_kernel!(eval, @Const(p), @Const(y), @Const(w))
    i = @index(Global, Linear)
    K = size(p, 1)
    @inbounds if i <= length(y)
        isum = zero(eltype(eval))
        for k in 1:K
            isum += exp(p[k, i])
        end
        eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
end

function EvoTrees.mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; kwargs...) where {T<:AbstractFloat}
    backend = get_backend(eval)
    eval_mlogloss_kernel!(backend)(eval, p, y, w; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end