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
    ϵ = eps(typeof(pk)(1e-7))
    return 2 * (yk * log(yk / pred + ϵ) + pred - yk)
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
            acc += _metric_value(M, p[k, i], _target(y, k, i), alpha)
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

@inline function _mle2p_metric_value(::Type{GaussianMLE}, loc, scale_raw, y)
    scale = exp(scale_raw)
    return -(log(scale) + (y - loc)^2 / (2 * scale^2))
end
@inline function _mle2p_metric_value(::Type{LogisticMLE}, loc, scale_raw, y)
    scale = exp(scale_raw)
    z = (y - loc) / scale
    az = abs(z)
    return -log(scale) - az - 2 * log1p(exp(-az))
end

function _eval_mle2p_metric(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T},
    ::Type{M};
    kwargs...
) where {T,M<:MLE2P}
    Y = size(p, 1) ÷ 2
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for t in 1:Y
            acc += _mle2p_metric_value(M, p[2t-1, i], p[2t, i], _target(y, t, i))
        end
        eval[i] = w[i] * acc / Y
    end
    return sum(Float64, eval) / sum(Float64, w)
end

gaussian_mle(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_mle2p_metric(p, y, w, eval, GaussianMLE; kwargs...)
logistic_mle(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T} =
    _eval_mle2p_metric(p, y, w, eval, LogisticMLE; kwargs...)

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


# NDCG within a single group, `pred` and `rel` in matching order.
function _ndcg_group(pred::AbstractVector, rel::AbstractVector, k::Int)
    n = length(rel)
    kk = min(k, n)
    ord = sortperm(pred; rev=true)
    dcg = 0.0
    @inbounds for i in 1:kk
        dcg += (2.0^rel[ord[i]] - 1) / log2(i + 1)
    end
    ideal = sort(rel; rev=true)
    idcg = 0.0
    @inbounds for i in 1:kk
        idcg += (2.0^ideal[i] - 1) / log2(i + 1)
    end
    # All-irrelevant groups score 1.0, matching the convention in the LTRC tutorial.
    return idcg > 0 ? dcg / idcg : 1.0
end

"""
    ndcg(p, y, w, eval; group, ndcg_k, kwargs...)

Normalised discounted cumulative gain, computed within each group then averaged over groups.
Requires the group index supplied at fit through `group_name` or `group_eval`. A group's weight
is the mean of its rows' weights, so the default of unit weights leaves every group equally
weighted. Only that group-level weight enters the score: NDCG is defined from the ranking of a
group's documents, so the spread of weights within a group is deliberately ignored. Use `:corr`
if per-document weights need to count.
"""
function ndcg(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector{T},
    eval::AbstractVector{T};
    group=nothing,
    ndcg_k::Int=typemax(Int),
    kwargs...
) where {T}
    isnothing(group) && error(
        "`metric = :ndcg` requires group information. Pass `group_name` when fitting from a " *
        "table, or `group_eval` alongside `x_eval` when fitting from a matrix."
    )
    ng = ngroups(group)
    scores = zeros(Float64, ng)
    weights = zeros(Float64, ng)
    @threads for g in 1:ng
        rows = group_rows(group, g)
        pred = [p[1, r] for r in rows]
        rel = [y[r] for r in rows]
        scores[g] = _ndcg_group(pred, rel, ndcg_k)
        weights[g] = mean(w[r] for r in rows)
    end
    return sum(scores .* weights) / sum(weights)
end

function _corr_group(pred::AbstractVector, obs::AbstractVector, wt::AbstractVector)
    length(pred) < 2 && return nothing
    sw = sum(wt)
    sw <= 0 && return nothing
    mp = sum(wt .* pred) / sw
    mo = sum(wt .* obs) / sw
    vp = sum(wt .* (pred .- mp) .^ 2) / sw
    vo = sum(wt .* (obs .- mo) .^ 2) / sw
    # A constant target carries nothing to correlate against, so the group is left out.
    vo <= 0 && return nothing
    # A constant prediction is a failure to discriminate, which scores as no correlation.
    vp <= 0 && return 0.0
    return sum(wt .* (pred .- mp) .* (obs .- mo)) / sw / sqrt(vp * vo)
end

"""
    corr(p, y, w, eval; group, kwargs...)

Weighted Pearson correlation between prediction and target, computed within each group then
averaged over groups. Requires the group index supplied at fit through `group_name`,
`eval_group_name` or `group_eval`. Unlike `:ndcg` this uses weights at both levels: a row's
weight enters its own group's correlation, and a group weighs by the mean of its rows' weights.

Groups of fewer than two rows, and groups whose target is constant, carry no signal and are
left out of the average. A group whose prediction is constant while its target is not scores
zero.
"""
function corr(
    p::AbstractMatrix{T},
    y::AbstractVector,
    w::AbstractVector{T},
    eval::AbstractVector{T};
    group=nothing,
    kwargs...
) where {T}
    isnothing(group) && error(
        "`metric = :corr` requires group information. Pass `group_name` or `eval_group_name` " *
        "when fitting from a table, or `group_eval` alongside `x_eval` when fitting from a matrix."
    )
    ng = ngroups(group)
    scores = zeros(Float64, ng)
    weights = zeros(Float64, ng)
    @threads for g in 1:ng
        rows = group_rows(group, g)
        pred = [Float64(p[1, r]) for r in rows]
        obs = [Float64(y[r]) for r in rows]
        wt = [Float64(w[r]) for r in rows]
        s = _corr_group(pred, obs, wt)
        if !isnothing(s)
            scores[g] = s
            weights[g] = mean(wt)
        end
    end
    sw = sum(weights)
    sw <= 0 && return zero(Float64)
    return sum(scores .* weights) / sw
end

function gini_raw(p::AbstractVector, y::AbstractVector)
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
    return gini_raw(p, y) / gini_raw(y, y)
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
    :ndcg => ndcg,
    :corr => corr,
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
is_maximise(::typeof(ndcg)) = true
is_maximise(::typeof(corr)) = true