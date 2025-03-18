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

# Converts MSE -> :mse
# const _type2loss_dict = Dict(
#     MSE => :mse,
#     LogLoss => :logloss,
#     Poisson => :poisson,
#     Gamma => :gamma,
#     Tweedie => :tweedie,
#     MLogLoss => :mlogloss,
#     GaussianMLE => :gaussian_mle,
#     LogisticMLE => :logistic_mle,
#     Quantile => :quantile,
#     MAE => :mae,
#     CredVar => :cred_var,
#     CredStd => :cred_std
# )
# _type2loss(L::Type) = _type2loss_dict[L]

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

# Credibility-based
function update_grads!(∇::Matrix, p::Matrix{T}, y::Vector{T}, ::Type{<:Cred}, params::EvoTypes) where {T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        @inbounds ∇[2, i] = (y[i] - p[1, i])^2 * ∇[3, i]
    end
end

# MSE
function update_grads!(∇::Vector{G}, p::Matrix{T}, y::Vector{T}, ::Type{MSE}, params::EvoTypes) where {G,T}
    @threads for i in eachindex(y)
        ∇[i] = G(
            2 * (p[1, i] - y[i]) * ∇[i][3],
            2 * ∇[i][3],
            ∇[i][3]
        )
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
function get_gain(::Type{L}, params::EvoTypes, ∑) where {L<:GradientRegression}
    # ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    ∑[1]^2 / (∑[2] + lambda * ∑[3] + L2) / 2
end

# GaussianRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:MLE2P,T}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    (∑[1]^2 / max(ϵ, (∑[3] + lambda * ∑[5] + L2)) + ∑[2]^2 / max(ϵ, (∑[4] + lambda * ∑[5] + L2))) / 2
end

# MultiClassRegression
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:MLogLoss,T}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    gain = zero(T)
    K = (length(∑) - 1) ÷ 2
    @inbounds for k = 1:K
        gain += ∑[k]^2 / max(ϵ, (∑[k+K] + lambda * ∑[end] + L2)) / 2
    end
    return gain
end

# MAE
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:MAE,T}
    ϵ = eps(T)
    abs(∑[1]) / max(ϵ, (1 + params.lambda + params.L2 / ∑[3]))
end

# Quantile
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:Quantile,T}
    ϵ = eps(T)
    abs(∑[1]) / max(ϵ, (1 + params.lambda + params.L2 / ∑[3]))
end

# CredVar: ratio of variance
# VHM = E²[X] = (∑1 / ∑3)²
# EVPV = E[X^2] - E²[X] = ∑2 / ∑3 - VHM
@inline function _get_cred(::Type{CredVar}, params::EvoTypes, ∑::AbstractVector{T}) where {T}
    ϵ = eps(eltype(∑))
    VHM = (∑[1] / ∑[3])^2
    EVPV = max(ϵ, (∑[2] / ∑[3] - VHM) / (1 + params.lambda * ∑[3]))
    return VHM / (VHM + EVPV + params.L2 / ∑[3])
end

# CredStd: ratio of std dev 
# VHM = E²[X] = (∑1 / ∑3)²
# EVPV = E[X^2] - E²[X] = ∑2 / ∑3 - VHM
@inline function _get_cred(::Type{CredStd}, params::EvoTypes, ∑::AbstractVector{T}) where {T}
    ϵ = eps(eltype(∑))
    VHM = (∑[1] / ∑[3])^2
    EVPV = max(ϵ, (∑[2] / ∑[3] - VHM) / (1 + params.lambda * ∑[3]))
    return sqrt(VHM) / (sqrt(VHM) + sqrt(EVPV) + sqrt(params.L2 / ∑[3]))
end

# gain for Cred
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:Cred,T}
    Z = _get_cred(L, params, ∑)
    return Z * abs(∑[1])
end
