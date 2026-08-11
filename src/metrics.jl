@inline _metric_target(y::AbstractVector, k, i) = y[i]
@inline _metric_target(y::AbstractMatrix, k, i) = y[k, i]

@inline _metric_value(::Type{MSE}, pk, yk, alpha) = (pk - yk)^2
@inline _metric_value(::Type{MAE}, pk, yk, alpha) = abs(pk - yk)

@inline function _metric_value(::Type{Quantile}, pk, yk, alpha)
    return alpha * max(yk - pk, zero(pk)) + (1 - alpha) * max(pk - yk, zero(pk))
end

@inline function _metric_value(::Type{LogLoss}, pk, yk, alpha)
    pred = sigmoid(pk)
    return -yk * log(pred) + (yk - 1) * log(1 - pred)
end

@inline function _metric_value(::Type{Poisson}, pk, yk, alpha)
    pred = exp(pk)
    return 2 * (yk * (log(yk) - log(pred)) + pred - yk)
end

@inline function _metric_value(::Type{Gamma}, pk, yk, alpha)
    pred = exp(pk)
    return 2 * (log(pred / yk) + yk / pred - 1)
end

@inline function _metric_value(::Type{Tweedie}, pk, yk, alpha)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return 2 * (yk^(2 - rho) / (1 - rho) / (2 - rho) - yk * pred^(1 - rho) / (1 - rho) + pred^(2 - rho) / (2 - rho))
end

function _eval_metric(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T},
    ::Type{M};
    alpha=0.5,
    kwargs...
) where {T,M}
    K = size(p, 1)
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for k in 1:K
            acc += _metric_value(M, p[k, i], _metric_target(y, k, i), alpha)
        end
        eval[i] = w[i] * acc / K
    end
    return sum(Float64, eval) / sum(Float64, w)
end

mse(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, MSE; kwargs...)
rmse(p::AbstractMatrix{T}, y::AbstractVecOrMat, w::AbstractVector, eval::AbstractVector; kwargs...) where {T} =
    sqrt(mse(p, y, w, eval::AbstractVector; kwargs...))
mae(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, MAE; kwargs...)
logloss(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, LogLoss; kwargs...)

function mlogloss(
    p::AbstractMatrix{T},
    y::AbstractVector{<:Integer},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    K = size(p, 1)
    @threads for i in eachindex(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return sum(Float64, eval) / sum(Float64, w)
end

poisson(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, Poisson; kwargs...)
gamma(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, Gamma; kwargs...)
tweedie(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, Tweedie; kwargs...)
wmae(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_metric(p, y, w, eval, Quantile; kwargs...)

@inline _mle2p_metric_value(::Type{GaussianMLE}, μ, ls, yt) = -(ls + (yt - μ)^2 / (2 * exp(2 * ls)))
@inline _mle2p_metric_value(::Type{LogisticMLE}, μ, ls, yt) = log(1 / 4 * sech(exp(-ls) * (yt - μ))^2) - ls

_student_const(ν) = loggamma((ν + 1) / 2) - loggamma(ν / 2) - 0.5 * log(ν * π)
@inline function _mle2p_metric_value(::Type{StudentMLE}, μ, ls, yt, ν)
    u = (yt - μ)^2 * exp(-2 * ls)
    return _student_const(ν) - ls - (ν + 1) / 2 * log1p(u / ν)
end
@inline _mle2p_metric_value(::Type{GaussianMLE}, μ, ls, yt, _) = _mle2p_metric_value(GaussianMLE, μ, ls, yt)
@inline _mle2p_metric_value(::Type{LogisticMLE}, μ, ls, yt, _) = _mle2p_metric_value(LogisticMLE, μ, ls, yt)

function _eval_mle2p_metric(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T},
    ::Type{M};
    nu=4.0,
    kwargs...
) where {T,M<:MLE2P}
    Y = size(p, 1) ÷ 2
    ν = T(nu)
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for t in 1:Y
            acc += _mle2p_metric_value(M, p[2t-1, i], p[2t, i], _metric_target(y, t, i), ν)
        end
        eval[i] = w[i] * acc / Y
    end
    return sum(Float64, eval) / sum(Float64, w)
end

gaussian_mle(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_mle2p_metric(p, y, w, eval, GaussianMLE; kwargs...)
logistic_mle(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_mle2p_metric(p, y, w, eval, LogisticMLE; kwargs...)
student_mle(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_mle2p_metric(p, y, w, eval, StudentMLE; kwargs...)

function multiquantile(
    p::AbstractMatrix{T},
    y::AbstractVector{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    alphas,
    kwargs...
) where {T}
    K = length(alphas)
    @assert size(p, 1) == K
    @threads for i in eachindex(y)
        yi = y[i]
        wi = w[i]
        acc = zero(T)
        @inbounds for k in 1:K
            alpha = alphas[k]
            acc += _metric_value(Quantile, p[k, i], yi, alpha)
        end
        eval[i] = wi * acc / K
    end
    return sum(Float64, eval) / sum(Float64, w)
end


function gini_raw(p::V, y::V) where {V<:AbstractVector}
    _y = y .- minimum(y)
    if length(_y) < 2
        return 0.0
    end
    random = cumsum(ones(length(p)) ./ length(p)^2)
    y_sort = _y[sortperm(p)]
    y_cum = cumsum(y_sort) ./ sum(_y) ./ length(p)
    gini = sum(Float64, random .- y_cum)
    return gini
end

function gini_norm(p::AbstractVector, y::AbstractVector)
    if length(y) < 2
        return 0.0
    end
    return gini_raw(y, p) / gini_raw(y, y)
end

function gini(
    p::AbstractMatrix{T},
    y::AbstractVector{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return gini_norm(view(p, 1, :), y)
end

const metric_dict = Dict(
    :mse => mse,
    :rmse => rmse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :poisson_deviance => poisson,
    :poisson => poisson,
    :gamma_deviance => gamma,
    :gamma => gamma,
    :tweedie_deviance => tweedie,
    :tweedie => tweedie,
    :gaussian_mle => gaussian_mle,
    :gaussian => gaussian_mle,
    :logistic_mle => logistic_mle,
    :wmae => wmae,
    :quantile => wmae,
    :multiquantile => multiquantile,
    :gini => gini,
    :student_mle => student_mle,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(rmse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(poisson)) = false
is_maximise(::typeof(gamma)) = false
is_maximise(::typeof(tweedie)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(logistic_mle)) = true
is_maximise(::typeof(wmae)) = false
is_maximise(::typeof(multiquantile)) = false
is_maximise(::typeof(gini)) = true
is_maximise(::typeof(student_mle)) = true