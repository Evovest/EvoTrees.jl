function mse(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...,
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
    kwargs...,
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
    kwargs...,
) where {T}
    eval = zero(T)
    p_prob = exp.(p) ./ sum(exp.(p), dims = 1)
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
    kwargs...,
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
    kwargs...,
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
    kwargs...,
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
    kwargs...,
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    eval /= sum(w)
    return eval
end

function logistic_mle(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {T}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += -w[i] * (log(1 / 4 * sech(exp(-p[2, i]) * (y[i] - p[1, i]))^2) - p[2, i])
    end
    eval /= sum(w)
    return eval
end

function wmae(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector;
    alpha = 0.5,
    kwargs...,
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

struct CallBack{F,M,V,Y}
    feval::F
    x::M
    p::M
    y::Y
    w::V
end
function (cb::CallBack)(logger, iter, tree)
    predict!(cb.p, tree, cb.x)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return nothing
end

function CallBack(
    params::EvoTypes{L,K,T};
    metric,
    x_eval,
    y_eval,
    w_eval = nothing,
    offset_eval = nothing,
) where {L,K,T}
    feval = metric_dict[metric]
    x = convert(Matrix{T}, x_eval)
    p = zeros(T, K, length(y_eval))
    if L == Softmax
        if eltype(y_eval) <: CategoricalValue
            levels = CategoricalArrays.levels(y_eval)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(y_eval))
        else
            levels = sort(unique(y_eval))
            yc = CategoricalVector(y_eval, levels = levels)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    else
        y = convert(Vector{T}, y_eval)
    end
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)

    if !isnothing(offset_eval)
        L == Logistic && (offset_eval .= logit.(offset_eval))
        L in [Poisson, Gamma, Tweedie] && (offset_eval .= log.(offset_eval))
        L == Softmax && (offset_eval .= log.(offset_eval))
        L in [GaussianMLE, LogisticMLE] && (offset_eval[:, 2] .= log.(offset_eval[:, 2]))
        offset_eval = T.(offset_eval)
        p .+= offset_eval'
    end

    if params.device == "gpu"
        return CallBack(feval, CuArray(x), CuArray(p), CuArray(y), CuArray(w))
    else
        return CallBack(feval, x, p, y, w)
    end
end

function init_logger(; T, metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :iter => Int[],
        :metrics => T[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger, iter, metric)
    logger[:nrounds] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    if iter == 0
        logger[:best_metric] = metric
    else
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end