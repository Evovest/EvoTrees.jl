# MSE
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{MSE})
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
end

# LogLoss - on linear predictor
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{LogLoss})
    @threads for i in eachindex(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
end

# Poisson
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{Poisson})
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
end

# Gamma
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{Gamma})
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
end

# Tweedie
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{Tweedie})
    rho = eltype(p)(1.5)
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
end

# MLogLoss
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{MLogLoss})
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

# L1
# function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{L1})
#     @threads for i in eachindex(y)
#         @inbounds ∇[1, i] =
#             (params.alpha * max(y[i] - p[1, i], 0) - (1 - params.alpha) * max(p[1, i] - y[i], 0)) *
#             ∇[3, i]
#     end
# end

# # Quantile
# function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{Quantile})
#     @threads for i in eachindex(y)
#         @inbounds ∇[1, i] = y[i] > p[1, i] ? params.alpha * ∇[3, i] : (params.alpha - 1) * ∇[3, i]
#         @inbounds ∇[2, i] = y[i] - p[1, i] # δ² serves to calculate the quantile value - hence no weighting on δ²
#     end
# end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{GaussianMLE})
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
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Type{LogisticMLE})
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
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:GradientRegression,T}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    ∑[1]^2 / max(ϵ, (∑[2] + lambda * ∑[3] + L2)) / 2
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

# Quantile
function get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L<:Quantile,T}
    abs(∑[1])
end
