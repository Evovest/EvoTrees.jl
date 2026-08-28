abstract type LossType end
abstract type GradientRegression <: LossType end
abstract type MLE2P <: LossType end # 2-parameters max-likelihood

abstract type MSE <: GradientRegression end
abstract type LogLoss <: GradientRegression end
abstract type Poisson <: GradientRegression end
abstract type Gamma <: GradientRegression end
abstract type Tweedie <: GradientRegression end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: MLE2P end
abstract type LogisticMLE <: MLE2P end
abstract type Quantile <: LossType end
abstract type MultiQuantile <: Quantile end
abstract type MAE <: LossType end
abstract type Cred <: LossType end
abstract type CredVar <: Cred end
abstract type CredStd <: Cred end

const _loss2type_dict = Dict(
    :mse => MSE,
    :logloss => LogLoss,
    :poisson => Poisson,
    :gamma => Gamma,
    :tweedie => Tweedie,
    :mlogloss => MLogLoss,
    :gaussian_mle => GaussianMLE,
    :logistic_mle => LogisticMLE,
    :quantile => Quantile,
    :multiquantile => MultiQuantile,
    :mae => MAE,
    :cred_var => CredVar,
    :cred_std => CredStd
)

################################################################################
# Backend-agnostic scalar math.
#
# Everything below is scalars-in / scalars-out, allocation-free and `@inline`d.
# It is called both by the threaded CPU `update_grads!` loops in this file and
# by the KernelAbstractions kernels in `ext/EvoTreesKernelAbstractionsExt/`, so
# that each loss expression has exactly one definition in the package.
#
# `@propagate_inbounds` (rather than plain `@inline`) on the accessors is load
# bearing: without it the caller's `@inbounds` does not reach these bodies, and
# the bounds check survives into device code as a `throw` branch.
################################################################################

Base.@propagate_inbounds _target(y::AbstractVector, k, i) = y[i]
Base.@propagate_inbounds _target(y::AbstractMatrix, k, i) = y[k, i]

@inline gradreg_grad_hess(::Type{MSE}, pk, yk) = (2 * (pk - yk), 2 * one(pk))

@inline function gradreg_grad_hess(::Type{LogLoss}, pk, yk)
    pred = sigmoid(pk)
    return (pred - yk, pred * (1 - pred))
end

@inline function gradreg_grad_hess(::Type{Poisson}, pk, yk)
    pred = exp(pk)
    return (pred - yk, pred)
end

@inline function gradreg_grad_hess(::Type{Gamma}, pk, yk)
    pred = exp(pk)
    return (2 * (1 - yk / pred), 2 * yk / pred)
end

@inline function gradreg_grad_hess(::Type{Tweedie}, pk, yk)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return (
        2 * (pred^(2 - rho) - yk * pred^(1 - rho)),
        2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * yk * pred^(1 - rho)),
    )
end

@inline mae_grad_hess(pk, yk) = (yk - pk, zero(pk))

@inline function cred_grad_hess(pk, yk)
    d = yk - pk
    return (d, d^2)
end

# Quantile keeps the raw residual in the hessian slot -- it is consumed later
# when computing leaf quantiles -- so the second element is deliberately *not*
# weight-scaled by the caller.
@inline function quantile_grad_diff(pk, yk, alpha)
    diff = yk - pk
    return (diff > 0 ? alpha : alpha - one(alpha), diff)
end

# MLogLoss. `isum` is the softmax denominator for observation `i`; it spans all
# K classes and so is accumulated by the caller before this is applied per-class.
@inline function mlogloss_grad_hess(pk, isum, is_target)
    prob = exp(pk) / isum
    return (is_target ? prob - one(prob) : prob, prob * (1 - prob))
end

function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{L}, params::EvoTypes) where {T,L<:GradientRegression}
    K = size(p, 1)
    w_row = 2 * K + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, h = gradreg_grad_hess(L, p[k, i], _target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
end

# MLogLoss
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector, ::Type{MLogLoss}, params::EvoTypes) where {T}
    K = size(p, 1)
    @threads for i in eachindex(y)
        isum = zero(T)
        @inbounds for k = 1:K
            isum += exp(p[k, i])
        end
        @inbounds w = ∇[end, i]
        @inbounds for k = 1:K
            g, h = mlogloss_grad_hess(p[k, i], isum, k == y[i])
            ∇[k, i] = g * w
            ∇[k+K, i] = h * w
        end
    end
end

# MAE
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{MAE}, params::EvoTypes) where {T}
    K = size(p, 1)
    w_row = 2 * K + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, h = mae_grad_hess(p[k, i], _target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
end

# Quantile
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{Quantile}, params::EvoTypes) where {T}
    K = size(p, 1)
    w_row = 2 * K + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, diff = quantile_grad_diff(p[k, i], _target(y, k, i), params.alpha)
            ∇[k, i] = g * w
            ∇[K+k, i] = diff
        end
    end
end

# MultiQuantile
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{MultiQuantile}, params::EvoTypes,) where {T}
    K = length(params.alphas)
    w_idx = 2 * K + 1
    @threads for i in eachindex(y)
        yi = y[i]
        @inbounds wi = ∇[w_idx, i]
        @inbounds for k in 1:K
            g, diff = quantile_grad_diff(p[k, i], yi, params.alphas[k])
            ∇[k, i] = g * wi
            ∇[K + k, i] = diff
        end
    end
end

# Credibility-based
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{<:Cred}, params::EvoTypes) where {T}
    K = size(p, 1)
    w_row = 2 * K + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, h = cred_grad_hess(p[k, i], _target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
end

# Two-parameter MLE. The tree predicts a location and an unconstrained scale
# parameter; the positive scale is `exp(scale_raw)`. Second-order terms
# are Fisher information (expected Hessian). Location and scale are orthogonal,
# so the Fisher matrix is diagonal and positive definite.

# Gaussian N(loc, scale²) — http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# `dscale` is d(exp)/d(scale_raw) = exp(scale_raw) = scale, so dscale/scale = 1.
@inline function mle2p_grad_hess(::Type{GaussianMLE}, loc, scale_raw, y)
    scale = exp(scale_raw)
    resid = loc - y
    g_loc = resid / scale^2
    g_scale = 1 - resid^2 / scale^2
    h_loc = 1 / scale^2
    h_scale = oftype(scale, 2)
    return (g_loc, g_scale, h_loc, h_scale)
end

# Logistic(loc, scale) — https://en.wikipedia.org/wiki/Logistic_distribution
# `dscale` is d(exp)/d(scale_raw) = scale, so dscale/scale = 1.
@inline function mle2p_grad_hess(::Type{LogisticMLE}, loc, scale_raw, y)
    scale = exp(scale_raw)
    z = (y - loc) / scale
    th = tanh(z / 2)
    g_loc = -th / scale
    g_scale = 1 - z * th
    h_loc = 1 / (3 * scale^2)
    h_scale = (oftype(scale, π)^2 + 3) / 9
    return (g_loc, g_scale, h_loc, h_scale)
end

function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{L}, params::EvoTypes) where {T,L<:MLE2P}
    Y = size(p, 1) ÷ 2
    w_row = 4 * Y + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            loc = p[2t-1, i]
            scale_raw = p[2t, i]
            g1, g2, h1, h2 = mle2p_grad_hess(L, loc, scale_raw, _target(y, t, i))
            ∇[gb+1, i] = g1 * w
            ∇[gb+2, i] = g2 * w
            ∇[hb+1, i] = h1 * w
            ∇[hb+2, i] = h2 * w
        end
    end
end

# utility functions
function logit(x::AbstractArray{T}) where {T<:AbstractFloat}
    return logit.(x)
end
@inline function logit(x::T) where {T<:AbstractFloat}
    @fastmath log(x / (1 - x))
end

function sigmoid(x::AbstractArray{T}) where {T<:AbstractFloat}
    return sigmoid.(x)
end
@inline function sigmoid(x::T) where {T<:AbstractFloat}
    @fastmath 1 / (1 + exp(-x))
end

@inline function unconstrain_mle_scale!(offset::AbstractArray)
    @views offset[:, 2:2:end] .= log.(offset[:, 2:2:end])
    return offset
end

# CredVar: ratio of variance
# VHM = E²[X] = (m1 / w)²
# EVPV = E[X^2] - E²[X] = m2 / w - VHM
@inline function _cred_Z(::Type{CredVar}, m1, m2, w, ϵ)
    VHM = (m1 / w)^2
    EVPV = max(ϵ, m2 / w - VHM)
    return VHM / (VHM + EVPV)
end

# CredStd: ratio of std dev
@inline function _cred_Z(::Type{CredStd}, m1, m2, w, ϵ)
    VHM = (m1 / w)^2
    EVPV = max(ϵ, m2 / w - VHM)
    return sqrt(VHM) / (sqrt(VHM) + sqrt(EVPV))
end
