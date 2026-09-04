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
# The chunk bodies live in their own functions so the `@threads` closure does not box the
# captured arrays, which otherwise makes every per-group call a dynamic dispatch.
function _ndcg_chunk!(scores, weights, p, y, w, group, chunk, ndcg_k::Int)
    pred = Float64[]
    rel = Float64[]
    ord = Int[]
    ideal = Float64[]
    for g in chunk
        rows = group_rows(group, g)
        n = length(rows)
        resize!(pred, n)
        resize!(rel, n)
        sw = 0.0
        @inbounds for (i, r) in enumerate(rows)
            pred[i] = p[1, r]
            rel[i] = y[r]
            sw += w[r]
        end
        scores[g] = _ndcg_group!(ord, ideal, pred, rel, ndcg_k)
        weights[g] = sw / n
    end
    return nothing
end

function _corr_chunk!(scores, weights, p, y, w, group, chunk, K::Int)
    pred = Float64[]
    obs = Float64[]
    wt = Float64[]
    for g in chunk
        rows = group_rows(group, g)
        # each target correlates on its own, then the group takes their mean, matching
        # how the per-observation metrics average over K
        acc = 0.0
        scored = 0
        for k in 1:K
            s = _corr_group!(pred, obs, wt, p, y, w, rows, k)
            isnothing(s) && continue
            acc += s
            scored += 1
        end
        scored == 0 && continue
        scores[g] = acc / scored
        sw = 0.0
        @inbounds for r in rows
            sw += w[r]
        end
        weights[g] = sw / length(rows)
    end
    return nothing
end

# One chunk per thread, so per-group scratch can be allocated once per chunk rather than
# once per group. Chunks rather than `threadid()` because tasks may migrate between threads.
function _group_chunks(ng::Int)
    nt = min(Threads.nthreads(), max(ng, 1))
    per = cld(ng, nt)
    return [((c - 1) * per + 1):min(c * per, ng) for c in 1:nt if (c - 1) * per + 1 <= ng]
end

function _ndcg_group!(ord::Vector{Int}, ideal::Vector{Float64}, pred::AbstractVector, rel::AbstractVector, k::Int)
    n = length(rel)
    kk = min(k, n)
    resize!(ord, n)
    # QuickSort is in place; the default hybrid allocates scratch on every call
    sortperm!(ord, pred; rev=true, alg=QuickSort)
    dcg = 0.0
    @inbounds for i in 1:kk
        dcg += (2.0^rel[ord[i]] - 1) / log2(i + 1)
    end
    resize!(ideal, n)
    copyto!(ideal, rel)
    sort!(ideal; rev=true, alg=QuickSort)
    idcg = 0.0
    @inbounds for i in 1:kk
        idcg += (2.0^ideal[i] - 1) / log2(i + 1)
    end
    # All-irrelevant groups score 1.0, matching the convention in the LTRC tutorial.
    return idcg > 0 ? dcg / idcg : 1.0
end

_ndcg_group(pred::AbstractVector, rel::AbstractVector, k::Int) =
    _ndcg_group!(Vector{Int}(undef, length(rel)), Vector{Float64}(undef, length(rel)), pred, rel, k)

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
    # Groups are row ids rather than boundaries, so a group's rows need not be contiguous and
    # cannot be viewed. Scratch buffers are per chunk instead, so the gather does not allocate
    # once per group.
    @threads for chunk in _group_chunks(ng)
        _ndcg_chunk!(scores, weights, p, y, w, group, chunk, ndcg_k)
    end
    return sum(scores .* weights) / sum(weights)
end

# The rows of a group are scattered, so they are gathered once into contiguous scratch and
# the two moment passes then run over that rather than chasing the same scattered reads
# twice. The accumulator is Float64 regardless of `T`, because the centring cancels
# catastrophically in Float32 once predictions sit far from zero.
function _corr_group!(pred::Vector{Float64}, obs::Vector{Float64}, wt::Vector{Float64},
    p::AbstractMatrix, y, w::AbstractVector, rows, k::Int)
    n = length(rows)
    n < 2 && return nothing
    resize!(pred, n)
    resize!(obs, n)
    resize!(wt, n)
    # the gather is scattered and cannot vectorise, so it is kept apart from the arithmetic,
    # which then runs over contiguous scratch
    @inbounds for (i, r) in enumerate(rows)
        pred[i] = p[k, r]
        obs[i] = _target(y, k, r)
        wt[i] = w[r]
    end
    sw = 0.0
    mp = 0.0
    mo = 0.0
    @inbounds @simd for i in 1:n
        sw += wt[i]
        mp += wt[i] * pred[i]
        mo += wt[i] * obs[i]
    end
    sw <= 0 && return nothing
    mp /= sw
    mo /= sw
    cxy = 0.0
    vp = 0.0
    vo = 0.0
    @inbounds @simd for i in 1:n
        dp = pred[i] - mp
        do_ = obs[i] - mo
        cxy += wt[i] * dp * do_
        vp += wt[i] * dp * dp
        vo += wt[i] * do_ * do_
    end
    # A constant target carries nothing to correlate against, so the group is left out.
    vo <= 0 && return nothing
    # A constant prediction is a failure to discriminate, which scores as no correlation.
    vp <= 0 && return 0.0
    return cxy / sqrt(vp * vo)
end

_corr_group(p::AbstractMatrix, y, w::AbstractVector, rows, k::Int) =
    _corr_group!(Float64[], Float64[], Float64[], p, y, w, rows, k)


"""
    corr(p, y, w, eval; group, kwargs...)

Weighted Pearson correlation between prediction and target, computed within each group then
averaged over groups. Requires the group index supplied at fit through `group_name`,
`eval_group_name` or `group_eval`. Unlike `:ndcg` this uses weights at both levels: a row's
weight enters its own group's correlation, and a group weighs by the mean of its rows' weights.

Groups of fewer than two rows, and groups whose target is constant, carry no signal and are
left out of the average. A group whose prediction is constant while its target is not scores
zero. With multiple targets each is correlated on its own and the group takes their mean.
"""
function corr(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    group=nothing,
    kwargs...
) where {T}
    isnothing(group) && error(
        "`metric = :corr` requires group information. Pass `group_name` or `eval_group_name` " *
        "when fitting from a table, or `group_eval` alongside `x_eval` when fitting from a matrix."
    )
    # Number of targets, not of prediction rows: an MLE model carries its scale in row 2,
    # which is not something to correlate against the target.
    K = y isa AbstractMatrix ? size(y, 1) : 1
    ng = ngroups(group)
    scores = zeros(Float64, ng)
    weights = zeros(Float64, ng)
    @threads for chunk in _group_chunks(ng)
        _corr_chunk!(scores, weights, p, y, w, group, chunk, K)
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