# linear
function update_grads!(
    ∇::Matrix,
    p::Matrix,
    y::Vector,
    ::EvoTreeRegressor{L,T}
) where {L<:Linear,T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
end

# logistic - on linear predictor
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeRegressor{L,T}) where {L<:Logistic,T}
    @threads for i in eachindex(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
end

# Poisson
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeCount{L,T}) where {L<:Poisson,T}
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
end

# Gamma
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeRegressor{L,T}) where {L<:Gamma,T}
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
end

# Tweedie
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeRegressor{L,T}) where {L<:Tweedie,T}
    rho = eltype(p)(1.5)
    @threads for i in eachindex(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
end

# L1
function update_grads!(∇::Matrix, p::Matrix, y::Vector, params::EvoTreeRegressor{L,T}) where {L<:L1,T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] =
            (params.alpha * max(y[i] - p[1, i], 0) - (1 - params.alpha) * max(p[1, i] - y[i], 0)) *
            ∇[3, i]
    end
end

# Softmax
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeClassifier{L,T}) where {L<:Softmax,T}
    sums = sum(exp.(p), dims=1)
    K = (size(∇, 1) - 1) ÷ 2
    @threads for i in eachindex(y)
        @inbounds for k = 1:K
            # ∇[k, i] = (exp(p[k, i]) / sums[i] - (onehot(y[i], 1:K))) * ∇[2 * K + 1, i]
            if k == y[i]
                ∇[k, i] = (exp(p[k, i]) / sums[i] - 1) * ∇[2*K+1, i]
            else
                ∇[k, i] = (exp(p[k, i]) / sums[i]) * ∇[2*K+1, i]
            end
            ∇[k+K, i] = 1 / sums[i] * (1 - exp(p[k, i]) / sums[i]) * ∇[2*K+1, i]
        end
    end
end

# Quantile
function update_grads!(∇::Matrix, p::Matrix, y::Vector, params::EvoTreeRegressor{L,T}) where {L<:Quantile,T}
    @threads for i in eachindex(y)
        @inbounds ∇[1, i] = y[i] > p[1, i] ? params.alpha * ∇[3, i] : (params.alpha - 1) * ∇[3, i]
        @inbounds ∇[2, i] = y[i] - p[1, i] # δ² serves to calculate the quantile value - hence no weighting on δ²
    end
end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::Union{EvoTreeGaussian{L,T},EvoTreeMLE{L,T}}) where {L<:GaussianMLE,T}
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
function update_grads!(∇::Matrix, p::Matrix, y::Vector, ::EvoTreeMLE{L,T}) where {L<:LogisticMLE,T}
    ϵ = eltype(p)(2e-7)
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

function softmax(x::AbstractVector{T}) where {T<:AbstractFloat}
    x .-= maximum(x)
    x = exp.(x) ./ sum(exp.(x))
    return x
end


##############################
# get the gain metric
##############################
# GradientRegression
function get_gain(
    params::EvoTypes{L,T},
    ∑::AbstractVector{T},
) where {L<:GradientRegression,T}
    ∑[1]^2 / (∑[2] + params.lambda * ∑[3]) / 2
end

# GaussianRegression
function get_gain(params::EvoTypes{L,T}, ∑::AbstractVector{T}) where {L<:MLE2P,T}
    (∑[1]^2 / (∑[3] + params.lambda * ∑[5]) + ∑[2]^2 / (∑[4] + params.lambda * ∑[5])) / 2
end

# MultiClassRegression
function get_gain(
    params::EvoTypes{L,T},
    ∑::AbstractVector{T},
) where {L<:MultiClassRegression,T}
    gain = zero(T)
    K = (length(∑) - 1) ÷ 2
    @inbounds for k = 1:K
        gain += ∑[k]^2 / (∑[k+K] + params.lambda * ∑[2*K+1]) / 2
    end
    return gain
end

# QuantileRegression
function get_gain(
    params::EvoTypes{L,T},
    ∑::AbstractVector{T},
) where {L<:QuantileRegression,T}
    abs(∑[1])
end

# L1 Regression
function get_gain(params::EvoTypes{L,T}, ∑::AbstractVector{T}) where {L<:L1Regression,T}
    abs(∑[1])
end
