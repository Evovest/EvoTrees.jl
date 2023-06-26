function mse(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        eval[i] = w[i] * (p[1, i] - y[i])^2
    end
    return sum(eval) / sum(w)
end
rmse(p::AbstractMatrix{T}, y::AbstractVector, w::AbstractVector; kwargs...) where {T} =
    sqrt(mse(p, y, w))

function mae(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        eval[i] = w[i] * abs(p[1, i] - y[i])
    end
    return sum(eval) / sum(w)
end

function logloss(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        pred = sigmoid(p[1, i])
        eval[i] = w[i] * (-y[i] * log(pred) + (y[i] - 1) * log(1 - pred))
    end
    return sum(eval) / sum(w)
end

function mlogloss(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    K = size(p, 1)
    @threads :static for i in eachindex(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return sum(eval) / sum(w)
end

function poisson(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        pred = exp(p[1, i])
        eval[i] = w[i] * 2 * (y[i] * (log(y[i]) - log(pred)) + pred - y[i])
    end
    return sum(eval) / sum(w)
end

function gamma(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        pred = exp(p[1, i])
        eval[i] = w[i] * 2 * (log(pred / y[i]) + y[i] / pred - 1)
    end
    return sum(eval) / sum(w)
end

function tweedie(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    rho = T(1.5)
    @threads :static for i in eachindex(y)
        pred = exp(p[1, i])
        eval[i] =
            w[i] *
            2 *
            (
                y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * pred^(1 - rho) / (1 - rho) +
                pred^(2 - rho) / (2 - rho)
            )
    end
    return sum(eval) / sum(w)
end

function gaussian_mle(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        eval[i] = -w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return sum(eval) / sum(w)
end

function logistic_mle(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        eval[i] = w[i] * (log(1 / 4 * sech(exp(-p[2, i]) * (y[i] - p[1, i]))^2) - p[2, i])
    end
    return sum(eval) / sum(w)
end

function wmae(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
    alpha=0.5,
    kwargs...
) where {T}
    @threads :static for i in eachindex(y)
        eval[i] =
            w[i] * (
                alpha * max(y[i] - p[1, i], zero(T)) +
                (1 - alpha) * max(p[1, i] - y[i], zero(T))
            )
    end
    return sum(eval) / sum(w)
end


function gini_raw(p::AbstractVector, y::AbstractVector)
    _y = y .- minimum(y)
    if length(_y) < 2
        return 0.0
    end
    random = cumsum(ones(length(p)) ./ length(p)^2)
    y_sort = _y[sortperm(p)]
    y_cum = cumsum(y_sort) ./ sum(_y) ./ length(p)
    gini = sum(random .- y_cum)
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
    y::AbstractVector,
    w::AbstractVector,
    eval::AbstractVector;
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
    :gini => gini,
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
is_maximise(::typeof(gini)) = true