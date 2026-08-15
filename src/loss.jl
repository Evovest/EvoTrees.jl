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

@inline _target(y::AbstractVector, k, i) = y[i]
@inline _target(y::AbstractMatrix, k, i) = y[k, i]

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
        @inbounds for k = 1:K
            iexp = exp(p[k, i])
            if k == y[i]
                ∇[k, i] = (iexp / isum - 1) * ∇[end, i]
            else
                ∇[k, i] = iexp / isum * ∇[end, i]
            end
            ∇[k+K, i] = 1 / isum * (1 - iexp / isum) * ∇[end, i]
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
            ∇[k, i] = (_target(y, k, i) - p[k, i]) * w
            ∇[K+k, i] = zero(T)
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
            diff = _target(y, k, i) - p[k, i]
            ∇[k, i] = diff > 0 ? params.alpha * w : (params.alpha - 1) * w
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
            diff = yi - p[k, i]
            alpha = params.alphas[k]
            ∇[k, i] = diff > 0 ? alpha * wi : (alpha - 1) * wi
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
            d = _target(y, k, i) - p[k, i]
            ∇[k, i] = d * w
            ∇[K+k, i] = d^2 * w
        end
    end
end

# Two-parameter MLE: unconstrained scale φ with σ = softplus(φ) (s for Logistic).
# Second-order terms are Fisher information (expected Hessian), not the observed Hessian.
# Location-scale families are orthogonal, so the Fisher matrix is diagonal and PD.

# Gaussian N(μ, σ²) - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# I_μμ = 1/σ², I_σσ = 2/σ², I_φφ = I_σσ (σ')²
@inline function mle2p_grad_hess(::Type{GaussianMLE}, μ, φ, yt)
    σ = softplus(φ)
    σ′ = sigmoid(φ)
    invσ = 1 / σ
    invσ2 = invσ * invσ
    d = μ - yt
    z2 = d * d * invσ2
    gμ = d * invσ2
    gφ = (1 - z2) * σ′ * invσ
    hμ = invσ2
    hφ = 2 * (σ′ * invσ)^2
    return (gμ, gφ, hμ, hφ)
end

# Logistic(μ, s) - https://en.wikipedia.org/wiki/Logistic_distribution
# I_μμ = 1/(3s²), I_ss = (π² + 3)/(9s²), I_φφ = I_ss (s')²
@inline function mle2p_grad_hess(::Type{LogisticMLE}, μ, φ, yt)
    s = softplus(φ)
    s′ = sigmoid(φ)
    invs = 1 / s
    z = (yt - μ) * invs
    th = tanh(z / 2)
    gμ = -th * invs
    gφ = (1 - z * th) * s′ * invs
    hμ = invs * invs / 3
    hφ = (oftype(s, π)^2 + 3) / 9 * (s′ * invs)^2
    return (gμ, gφ, hμ, hφ)
end

function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::AbstractVecOrMat, ::Type{L}, params::EvoTypes) where {T,L<:MLE2P}
    Y = size(p, 1) ÷ 2
    w_row = 4 * Y + 1
    @threads for i in axes(p, 2)
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            φ = p[2t, i]
            g1, g2, h1, h2 = mle2p_grad_hess(L, μ, φ, _target(y, t, i))
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

# σ = log(1 + exp(φ)), overflow-safe
function softplus(x::AbstractArray{T}) where {T<:AbstractFloat}
    return softplus.(x)
end
@inline function softplus(x::T) where {T<:AbstractFloat}
    ax = abs(x)
    return max(x, zero(T)) + log1p(exp(-ax))
end

# inverse: φ = log(exp(σ) - 1), overflow-safe for σ > 0
function invsoftplus(x::AbstractArray{T}) where {T<:AbstractFloat}
    return invsoftplus.(x)
end
@inline function invsoftplus(x::T) where {T<:AbstractFloat}
    return x + log(-expm1(-x))
end

@inline function unconstrain_mle_scale!(offset::AbstractArray)
    @views offset[:, 2:2:end] .= invsoftplus.(offset[:, 2:2:end])
    return offset
end

##############################
# get the gain metric
##############################
# GradientRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:GradientRegression,T,V<:AbstractVector}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    K = (length(∑) - 1) ÷ 2
    gain = zero(T)
    @inbounds for k in 1:K
        gain += ∑L[k]^2 / max(ϵ, (∑L[k+K] + lambda * ∑L[end] + L2)) / 2
        gain += ∑R[k]^2 / max(ϵ, (∑R[k+K] + lambda * ∑R[end] + L2)) / 2
        gain -= ∑[k]^2  / max(ϵ, (∑[k+K]  + lambda * ∑[end]  + L2)) / 2
    end
    return gain
end

# GaussianRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:MLE2P,T,V<:AbstractVector}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    Y = (length(∑) - 1) ÷ 4
    wsum = ∑[end]
    wsumL = ∑L[end]
    wsumR = ∑R[end]
    gain = zero(T)
    @inbounds for t in 1:Y
        gb = 2 * (t - 1)
        hb = 2 * Y + 2 * (t - 1)
        for s in 1:2
            gain += ∑L[gb+s]^2 / max(ϵ, (∑L[hb+s] + lambda * wsumL + L2)) / 2
            gain += ∑R[gb+s]^2 / max(ϵ, (∑R[hb+s] + lambda * wsumR + L2)) / 2
            gain -= ∑[gb+s]^2 / max(ϵ, (∑[hb+s] + lambda * wsum + L2)) / 2
        end
    end
    return gain
end

# MultiClassRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:MLogLoss,T,V<:AbstractVector}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    gain = zero(T)
    K = (length(∑) - 1) ÷ 2
    @inbounds for k = 1:K
        gain += ∑L[k]^2 / max(ϵ, (∑L[k+K] + lambda * ∑L[end] + L2)) / 2
        gain += ∑R[k]^2 / max(ϵ, (∑R[k+K] + lambda * ∑R[end] + L2)) / 2
        gain -= ∑[k]^2 / max(ϵ, (∑[k+K] + lambda * ∑[end] + L2)) / 2
    end
    return gain
end

# MAE
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:MAE,T,V<:AbstractVector}
    ϵ = eps(T)
    K = (length(∑) - 1) ÷ 2
    w_p = ∑[end]
    w_l = ∑L[end]
    w_r = ∑R[end]
    d_l = max(ϵ, (1 + params.lambda + params.L2 / w_l))
    d_r = max(ϵ, (1 + params.lambda + params.L2 / w_r))
    gain = zero(T)
    @inbounds for k in 1:K
        gain += abs(∑L[k] / w_l - ∑[k] / w_p) * w_l / d_l
        gain += abs(∑R[k] / w_r - ∑[k] / w_p) * w_r / d_r
    end
    return gain
end

# Quantile
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:Quantile,T,V<:AbstractVector}
    ϵ = eps(T)
    K = (length(∑) - 1) ÷ 2
    w_p = ∑[end]
    w_l = ∑L[end]
    w_r = ∑R[end]
    d_l = max(ϵ, (1 + params.lambda + params.L2 / w_l))
    d_r = max(ϵ, (1 + params.lambda + params.L2 / w_r))
    gain = zero(T)
    @inbounds for k in 1:K
        gain += abs(∑L[k] / w_l - ∑[k] / w_p) * w_l / d_l
        gain += abs(∑R[k] / w_r - ∑[k] / w_p) * w_r / d_r
    end
    return gain
end

# MultiQuantile
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:MultiQuantile,T,V<:AbstractVector}
    ϵ = eps(T)
    K = (length(∑) - 1) ÷ 2
    w_p = ∑[end]
    w_l = ∑L[end]
    w_r = ∑R[end]
    d_l = max(ϵ, (1 + params.lambda + params.L2 / w_l))
    d_r = max(ϵ, (1 + params.lambda + params.L2 / w_r))
    gain = zero(T)
    @inbounds for k in 1:K
        gain += abs(∑L[k] / w_l - ∑[k] / w_p) * w_l / d_l
        gain += abs(∑R[k] / w_r - ∑[k] / w_p) * w_r / d_r
    end
    return gain
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

function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:Cred,T}
    ϵ = eps(T)
    K = (length(∑) - 1) ÷ 2
    w = ∑[end]
    g = zero(T)
    @inbounds for k in 1:K
        Z = _cred_Z(L, ∑[k], ∑[K+k], w, ϵ)
        g += Z * abs(∑[k]) / (1 + params.L2 / w)
    end
    return g
end

function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:Cred,T,V<:AbstractVector}
    ϵ = eps(T)
    K = (length(∑) - 1) ÷ 2
    w = ∑[end]
    w_l = ∑L[end]
    w_r = ∑R[end]
    gain = zero(T)
    @inbounds for k in 1:K
        Z = _cred_Z(L, ∑[k], ∑[K+k], w, ϵ)
        ZL = _cred_Z(L, ∑L[k], ∑L[K+k], w_l, ϵ)
        ZR = _cred_Z(L, ∑R[k], ∑R[K+k], w_r, ϵ)
        gain += ZL * abs(∑L[k]) / (1 + params.L2 / w_l)
        gain += ZR * abs(∑R[k]) / (1 + params.L2 / w_r)
        gain -= Z * abs(∑[k]) / (1 + params.L2 / w)
    end
    return gain
end
