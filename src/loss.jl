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
    :mae => MAE,
    :cred_var => CredVar,
    :cred_std => CredStd
)

# MSE
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{MSE}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
end

# LogLoss - on linear predictor
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{LogLoss}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
end

# Poisson
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{Poisson}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
end

# Gamma
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{Gamma}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
end

# Tweedie
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{Tweedie}, params::EvoTypes) where {T}
    rho = eltype(p)(1.5)
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
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
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{MAE}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
    end
end

# Quantile
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{Quantile}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        diff = (y[i] - p[1, i])
        @inbounds ∇[1, i] = diff > 0 ? params.alpha * ∇[3, i] : (params.alpha - 1) * ∇[3, i]
        @inbounds ∇[2, i] = diff
    end
end

# Credibility-based
function update_grads!(∇::Matrix, p::Matrix{T}, y::Vector{T}, ::Type{<:Cred}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        @inbounds ∇[2, i] = (y[i] - p[1, i])^2 * ∇[3, i]
    end
end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{GaussianMLE}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        # first order
        @inbounds ∇[1, i] = (p[1, i] - y[i]) / exp(2 * p[2, i]) * ∇[5, i]
        @inbounds ∇[2, i] = (1 - (p[1, i] - y[i])^2 / exp(2 * p[2, i])) * ∇[5, i]
        # second order
        @inbounds ∇[3, i] = ∇[5, i] / exp(2 * p[2, i])
        @inbounds ∇[4, i] = ∇[5, i] * 2 / exp(2 * p[2, i]) * (p[1, i] - y[i])^2
    end
end

# LogisticProb - https://en.wikipedia.org/wiki/Logistic_distribution
# pdf = 
# pred[i][1] = μ
# pred[i][2] = log(s)
function update_grads!(∇::Matrix{T}, p::Matrix{T}, y::Vector{T}, ::Type{LogisticMLE}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        # first order
        @inbounds ∇[1, i] =
            -tanh((y[i] - p[1, i]) / (2 * exp(p[2, i]))) * exp(-p[2, i]) * ∇[5, i]
        @inbounds ∇[2, i] =
            -(
                exp(-p[2, i]) *
                (y[i] - p[1, i]) *
                tanh((y[i] - p[1, i]) / (2 * exp(p[2, i]))) - 1
            ) * ∇[5, i]
        # second order
        @inbounds ∇[3, i] =
            sech((y[i] - p[1, i]) / (2 * exp(p[2, i])))^2 / (2 * exp(2 * p[2, i])) *
            ∇[5, i]
        @inbounds ∇[4, i] =
            (
                exp(-2 * p[2, i]) *
                (p[1, i] - y[i]) *
                (p[1, i] - y[i] + exp(p[2, i]) * sinh(exp(-p[2, i]) * (p[1, i] - y[i])))
            ) / (1 + cosh(exp(-p[2, i]) * (p[1, i] - y[i]))) * ∇[5, i]
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

##############################
# get the gain metric
##############################
# GradientRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:GradientRegression,T,V<:AbstractVector}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    gain = ∑L[1]^2 / max(ϵ, (∑L[2] + lambda * ∑L[3] + L2)) / 2 +
           ∑R[1]^2 / max(ϵ, (∑R[2] + lambda * ∑R[3] + L2)) / 2 -
           ∑[1]^2 / max(ϵ, (∑[2] + lambda * ∑[3] + L2)) / 2
    return gain
end

# GaussianRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:MLE2P,T,V<:AbstractVector}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    gain = (∑L[1]^2 / max(ϵ, (∑L[3] + lambda * ∑L[5] + L2)) + ∑L[2]^2 / max(ϵ, (∑L[4] + lambda * ∑L[5] + L2))) / 2 +
           (∑R[1]^2 / max(ϵ, (∑R[3] + lambda * ∑R[5] + L2)) + ∑R[2]^2 / max(ϵ, (∑R[4] + lambda * ∑R[5] + L2))) / 2 -
           (∑[1]^2 / max(ϵ, (∑[3] + lambda * ∑[5] + L2)) + ∑[2]^2 / max(ϵ, (∑[4] + lambda * ∑[5] + L2))) / 2
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
    gain = abs(∑L[1] / ∑L[3] - ∑[1] / ∑[3]) * ∑L[3] / max(ϵ, (1 + params.lambda + params.L2 / ∑L[3])) +
           abs(∑R[1] / ∑R[3] - ∑[1] / ∑[3]) * ∑R[3] / max(ϵ, (1 + params.lambda + params.L2 / ∑R[3]))
    return gain
end

# Quantile
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:Quantile,T,V<:AbstractVector}
    ϵ = eps(T)
    gain = abs(∑L[1] / ∑L[3] - ∑[1] / ∑[3]) * ∑L[3] / max(ϵ, (1 + params.lambda + params.L2 / ∑L[3])) +
           abs(∑R[1] / ∑R[3] - ∑[1] / ∑[3]) * ∑R[3] / max(ϵ, (1 + params.lambda + params.L2 / ∑R[3]))
    return gain
end

# CredVar: ratio of variance
# VHM = E²[X] = (∑1 / ∑3)²
# EVPV = E[X^2] - E²[X] = ∑2 / ∑3 - VHM
@inline function _get_cred(::Type{CredVar}, params::EvoTypes, ∑::AbstractVector{T}) where {T}
    ϵ = eps(eltype(∑))
    VHM = (∑[1] / ∑[3])^2
    EVPV = max(ϵ, (∑[2] / ∑[3] - VHM))
    return VHM / (VHM + EVPV)
end

# CredStd: ratio of std dev 
# VHM = E²[X] = (∑1 / ∑3)²
# EVPV = E[X^2] - E²[X] = ∑2 / ∑3 - VHM
@inline function _get_cred(::Type{CredStd}, params::EvoTypes, ∑::AbstractVector{T}) where {T}
    ϵ = eps(eltype(∑))
    VHM = (∑[1] / ∑[3])^2
    EVPV = max(ϵ, (∑[2] / ∑[3] - VHM))
    return sqrt(VHM) / (sqrt(VHM) + sqrt(EVPV))
end

# gain for Cred
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:Cred,T}
    Z = _get_cred(L, params, ∑)
    return Z * abs(∑[1]) / (1 + params.L2 / ∑[3])
end
# gain for Cred
function get_gain(::Type{L}, params::EvoTypes, ∑::Vector{T}, ∑L::V, ∑R::V) where {L<:Cred,T,V<:AbstractVector}
    Z = _get_cred(L, params, ∑)
    ZL = _get_cred(L, params, ∑L)
    ZR = _get_cred(L, params, ∑R)
    gain = ZL * abs(∑L[1]) / (1 + params.L2 / ∑L[3]) +
           ZR * abs(∑R[1]) / (1 + params.L2 / ∑R[3]) -
           Z * abs(∑[1]) / (1 + params.L2 / ∑[3])
    return gain
end
