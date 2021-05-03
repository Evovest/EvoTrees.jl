# linear
function update_grads!(::Linear, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds for i in eachindex(y)
        δ𝑤[1,i] = 2 * (p[1,i] - y[i]) * δ𝑤[3,i]
        δ𝑤[2,i] = 2 * δ𝑤[3,i]
    end
end

# logistic - on linear predictor
function update_grads!(::Logistic, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds for i in eachindex(y)
        δ𝑤[1,i] = (sigmoid(p[1,i]) * (1 - y[i]) - (1 - sigmoid(p[1,i])) * y[i]) * δ𝑤[3,i]
        δ𝑤[2,i] = sigmoid(p[1,i]) * (1 - sigmoid(p[1,i])) * δ𝑤[3,i]
    end
end

# Poisson
function update_grads!(::Poisson, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds for i in eachindex(y)
        δ𝑤[1,i] = (exp(p[1, i]) .- y[i]) * δ𝑤[3,i]
        δ𝑤[2,i] = exp(p[1, i]) * δ𝑤[3,i]
    end
end

# L1
function update_grads!(::L1, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds for i in eachindex(y)
        δ𝑤[1, i] = (α * max(y[i] - p[1,i], 0) - (1 - α) * max(p[1,i] - y[i], 0)) * δ𝑤[3, i]
    end
end

# Softmax
function update_grads!(::Softmax, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{S}, α::T) where {T <: AbstractFloat,S}
    p .= p .- maximum(p, dims=1)
    sums = sum(exp.(p), dims=1)
    # K = (size(δ𝑤, 1) - 1) ÷ 2
    K = size(p, 1)
    for i in eachindex(y)
        for k in 1:K
            # δ𝑤[k, i] = (exp(p[k, i]) / sums[i] - (onehot(y[i], 1:K))) * δ𝑤[2 * K + 1, i]
            if k == y[i]
                δ𝑤[k, i] = (exp(p[k, i]) / sums[i] - 1) * δ𝑤[2 * K + 1, i]
            else
                δ𝑤[k, i] = (exp(p[k, i]) / sums[i]) * δ𝑤[2 * K + 1, i]
            end
            δ𝑤[k + K, i] = 1 / sums[i] * (1 - exp(p[k, i]) / sums[i]) * δ𝑤[2 * K + 1, i]
        end
    end
end

# Quantile
function update_grads!(::Quantile, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds for i in eachindex(y)
        δ𝑤[1,i] = y[i] > p[1,i] ? α * δ𝑤[3,i] : (α - 1) * δ𝑤[3,i]
        δ𝑤[2,i] = y[i] - p[1,i] # δ² serves to calculate the quantile value - hence no weighting on δ²
    end
end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(::Gaussian, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}, α::T) where {T <: AbstractFloat}
    @inbounds @simd for i in eachindex(y)
        # first order
        δ𝑤[1,i] = (p[1,i] - y[i]) / max(1e-8, exp(2 * p[2,i])) * δ𝑤[5,i]
        δ𝑤[2,i] = (1 - (p[1,i] - y[i])^2 / max(1e-8, exp(2 * p[2,i]))) * δ𝑤[5,i]
        # second order
        δ𝑤[3,i] = δ𝑤[5,i] / max(1e-8, exp(2 * p[2,i]))
        δ𝑤[4,i] = 2 * δ𝑤[5,i] / max(1e-8, exp(2 * p[2,i])) * (p[1,i] - y[i])^2
    end
end

# utility functions
function logit(x::AbstractArray{T,1}) where T <: AbstractFloat
    @. x = log(x / (1 - x))
    return x
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T,1}) where T <: AbstractFloat
    @. x = 1 / (1 + exp(-x))
    return x
end

function sigmoid(x::T) where T <: AbstractFloat
    x = 1 / (1 + exp(-x))
    return x
end

function softmax(x::AbstractVector{T}) where T <: AbstractFloat
    x .-= maximum(x)
    x = exp.(x) ./ sum(exp.(x))
    return x
end


##############################
# get the gain metric
##############################
# GradientRegression
function get_gain(::S, ∑::Vector{T}, λ::T, K) where {S <: GradientRegression,T <: AbstractFloat}
    ∑[1]^2 / (∑[2] + λ * ∑[3]) / 2
end

# GaussianRegression
function get_gain(::S, ∑::Vector{T}, λ::T, K) where {S <: GaussianRegression,T <: AbstractFloat}
    (∑[1]^2 / (∑[3] + λ * ∑[5]) + ∑[2]^2 / (∑[4] + λ * ∑[5])) / 2
end

# MultiClassRegression
function get_gain(::S, ∑::Vector{T}, λ::T, K) where {S <: MultiClassRegression,T <: AbstractFloat}
    gain = zero(T)
    @inbounds for k in 1:K
        gain += ∑[k]^2 / (∑[k + K] + λ * ∑[2 * K + 1]) / 2
    end
    return gain
end

# QuantileRegression
function get_gain(::S, ∑::Vector{T}, λ::T, K) where {S <: QuantileRegression,T <: AbstractFloat}
    abs(∑[1])
end

# L1 Regression
function get_gain(::S, ∑::Vector{T}, λ::T, K) where {S <: L1Regression,T <: AbstractFloat}
    abs(∑[1])
end


function update_childs_∑!(::L, nodes, n, bin, feat, K) where {L <: Union{GradientRegression,QuantileRegression,L1Regression}}
    nodes[n << 1].∑ .= nodes[n].hL[feat][(4 * bin - 3):(4 * bin)]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[feat][(4 * bin - 3):(4 * bin)]
    return nothing
end

function update_childs_∑!(::L, nodes, n, bin, feat, K) where {L <: GaussianRegression}
    nodes[n << 1].∑ .= nodes[n].hL[feat][(8 * bin - 7):(8 * bin)]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[feat][(8 * bin - 7):(8 * bin)]
    return nothing
end

function update_childs_∑!(::L, nodes, n, bin, feat, K) where {L <: MultiClassRegression}
    stride =  Int(ceil((2 * K + 1)/4)*4)
    nodes[n << 1].∑ .= nodes[n].hL[feat][(stride * (bin - 1) + 1):(stride * bin)]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[feat][(stride * (bin - 1) + 1):(stride * bin)]
    return nothing
end