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
            acc += EvoTrees._metric_value(M, p[k, i], EvoTrees._target(y, k, i), alpha)
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
            acc += EvoTrees._metric_value(EvoTrees.Quantile, p[k, i], yi, alphas[k])
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
            acc += EvoTrees._mle2p_metric_value(M, p[2t-1, i], p[2t, i], EvoTrees._target(y, t, i))
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

# NDCG needs each group sorted by predicted score, which on device means a segmented sort
# over variable-length groups. It is an eval metric run once per round rather than a hot
# path, so predictions are brought to the host and scored with the CPU code.
function EvoTrees.ndcg(
    p::CuMatrix{T},
    y::CuVector,
    w::CuVector{T},
    eval::CuVector{T};
    group=nothing,
    ndcg_k::Int=typemax(Int),
    kwargs...
) where {T<:AbstractFloat}
    isnothing(group) && error(
        "`metric = :ndcg` requires group information. Pass `group_name` when fitting from a " *
        "table, or `group_eval` alongside `x_eval` when fitting from a matrix."
    )
    p_cpu = Array(p)
    y_cpu = Array(y)
    w_cpu = Array(w)
    return EvoTrees.ndcg(p_cpu, y_cpu, w_cpu, similar(w_cpu, 0); group, ndcg_k)
end

# Same reasoning as `ndcg` above: a per-group correlation reads its rows individually, so the
# arrays are brought to the host rather than scalar-indexed on device.
function EvoTrees.corr(
    p::CuMatrix{T},
    y::CuVector,
    w::CuVector{T},
    eval::CuVector{T};
    group=nothing,
    kwargs...
) where {T<:AbstractFloat}
    isnothing(group) && error(
        "`metric = :corr` requires group information. Pass `group_name` or `eval_group_name` " *
        "when fitting from a table, or `group_eval` alongside `x_eval` when fitting from a matrix."
    )
    p_cpu = Array(p)
    y_cpu = Array(y)
    w_cpu = Array(w)
    return EvoTrees.corr(p_cpu, y_cpu, w_cpu, similar(w_cpu, 0); group)
end
