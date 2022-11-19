function mse(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (p[1, i] - y[i])^2
    end
    eval /= sum(w)
    return eval
end
rmse(p::AbstractMatrix{T}, y::AbstractVector, w::AbstractVector; kwargs...) where {T} =
    sqrt(mse(p, y, w))

function mae(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * abs(p[1, i] - y[i])
    end
    eval /= sum(w)
    return eval
end

function logloss(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    kwargs...,
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        pred = sigmoid(p[1, i])
        eval -= w[i] * (y[i] * log(pred) + (1 - y[i]) * log(1 - pred))
    end
    eval /= sum(w)
    return eval
end

function mlogloss(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    p_prob = exp.(p) ./ sum(exp.(p), dims=1)
    @inbounds for i in eachindex(y)
        eval -= w[i] * log(p_prob[y[i], i])
    end
    eval /= sum(w)
    return eval
end

function poisson_deviance(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    ϵ = eps(T)
    @inbounds for i in eachindex(y)
        pred = exp(p[1, i])
        eval += w[i] * 2 * (y[i] * log(y[i] / pred + ϵ) + pred - y[i])
    end
    eval /= sum(w)
    return eval
end

function gamma_deviance(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        pred = exp(p[1, i])
        eval += w[i] * 2 * (log(pred / y[i]) + y[i] / pred - 1)
    end
    eval /= sum(w)
    return eval
end

function tweedie_deviance(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    rho = T(1.5)
    @inbounds for i in eachindex(y)
        pred = exp(p[1, i])
        eval +=
            w[i] *
            2 *
            (
                y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * pred^(1 - rho) / (1 - rho) +
                pred^(2 - rho) / (2 - rho)
            )
    end
    eval /= sum(w)
    return eval
end

function gaussian_mle(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval -= w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    eval /= sum(w)
    return eval
end

function logistic_mle(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (log(1 / 4 * sech(exp(-p[2, i]) * (y[i] - p[1, i]))^2) - p[2, i])
    end
    eval /= sum(w)
    return eval
end

function wmae(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    alpha=0.5,
    kwargs...
) where {T}
    eval = zero(T)
    for i in eachindex(y)
        eval +=
            w[i] * (
                alpha * max(y[i] - p[1, i], zero(T)) +
                (1 - alpha) * max(p[1, i] - y[i], zero(T))
            )
    end
    eval /= sum(w)
    return eval
end


function gini_raw(y::T, p::S) where {T,S}
    if length(y) < 2
        return 0.0
    end
    random = (1:length(p)) ./ length(p)
    l_sort = y[sortperm(p)]
    l_cum_w = cumsum(l_sort) ./ sum(y)
    gini = sum(l_cum_w .- random)
    return gini
end

function gini_norm(y::T, p::S) where {T,S}
    if length(y) < 2
        return 0.0
    end
    return gini_raw(y, p) / gini_raw(y, y)
end

function gini(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    kwargs...,
) where {T}
    return -gini_norm(y, view(p, 1, :))
end

const metric_dict = Dict(
    :mse => mse,
    :rmse => rmse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :poisson_deviance => poisson_deviance,
    :poisson => poisson_deviance,
    :gamma_deviance => gamma_deviance,
    :gamma => gamma_deviance,
    :tweedie_deviance => tweedie_deviance,
    :tweedie => tweedie_deviance,
    :gaussian_mle => gaussian_mle,
    :gaussian => gaussian_mle,
    :logistic_mle => logistic_mle,
    :wmae => wmae,
    :quantile => wmae,
    :gini => gini,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(rmse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(poisson_deviance)) = false
is_maximise(::typeof(gamma_deviance)) = false
is_maximise(::typeof(tweedie_deviance)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(logistic_mle)) = true
is_maximise(::typeof(wmae)) = false
is_maximise(::typeof(gini)) = true