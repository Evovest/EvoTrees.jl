# linear
function update_grads!(loss::Linear, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds for i in eachindex(δ)
        δ[i] = 2 .* (pred[i] .- target[i]) .* 𝑤[i]
        δ²[i] = 2 .* 𝑤[i]
    end
end

# logistic - on linear predictor
function update_grads!(loss::Logistic, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds for i in eachindex(δ)
        # δ[i] = (sigmoid.(pred[i]) .* (1 .- target[i]) .- (1 .- sigmoid.(pred[i])) .* target[i]) .* 𝑤[i]
        # δ²[i] = sigmoid.(pred[i]) .* (1 .- sigmoid.(pred[i])) .* 𝑤[i]
        δ[i] = (sigmoid(pred[i][1]) * (1 - target[i]) - (1 - sigmoid(pred[i][1])) * target[i][1]) * 𝑤[i]
        δ²[i] = sigmoid(pred[i][1]) * (1 - sigmoid(pred[i][1])) * 𝑤[i]
    end
end

# Poisson
function update_grads!(loss::Poisson, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds for i in eachindex(δ)
        δ[i] = (exp.(pred[i]) .- target[i]) .* 𝑤[i]
        δ²[i] = exp.(pred[i]) .* 𝑤[i]
    end
end

# L1
function update_grads!(loss::L1, α::T, pred::Vector{SVector{L,T}}, target::AbstractArray{T, 1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds for i in eachindex(δ)
        δ[i] =  (α * max(target[i] - pred[i][1], 0) - (1-α) * max(pred[i][1] - target[i], 0)) * 𝑤[i]
    end
end

# Softmax
function update_grads!(loss::Softmax, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{Int}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    pred = pred - maximum.(pred)
    # sums = sum(exp.(pred), dims=2)
    @inbounds for i in 1:size(pred,1)
        sums = sum(exp.(pred[i]))
        δ[i] = (exp.(pred[i]) ./ sums - (onehot(target[i], 1:L))) * 𝑤[i][1]
        δ²[i] =  1 / sums * (1 - exp.(pred[i]) ./ sums) * 𝑤[i][1]
    end
end

# Quantile
function update_grads!(loss::Quantile, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds for i in eachindex(δ)
        δ[i] = target[i] > pred[i][1] ? α * 𝑤[i] : (α - 1) * 𝑤[i]
        δ²[i] = target[i] - pred[i] # δ² serves to calculate the quantile value - hence no weighting on δ²
    end
end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
function update_grads!(loss::Gaussian, α, pred::Vector{SVector{L,T}}, target::AbstractArray{T, 1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    @inbounds @threads for i in eachindex(δ)
        δ[i] = SVector((pred[i][1] - target[i]) / exp(pred[i][2]) * 𝑤[i][1], 𝑤[i][1] / 2 * (1 - (pred[i][1] - target[i])^2 / exp(pred[i][2])))
        δ²[i] = SVector(𝑤[i][1] / exp(pred[i][2]), 𝑤[i][1] / exp(pred[i][2]) * (pred[i][1] - target[i])^2)
    end
end

# utility functions
function logit(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = log(x / (1 - x))
    return x
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T, 1}) where T <: AbstractFloat
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
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GradientRegression, T <: AbstractFloat, L}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end

# MultiClassRegression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: MultiClassRegression, T <: AbstractFloat, L}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end

# L1 Regression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: L1Regression, T <: AbstractFloat, L}
    gain = sum(abs.(∑δ))
    return gain
end

# QuantileRegression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: QuantileRegression, T <: AbstractFloat, L}
    gain = sum(abs.(∑δ) ./ (1 .+ λ))
    return gain
end

# GaussianRegression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GaussianRegression, T <: AbstractFloat, L}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end
