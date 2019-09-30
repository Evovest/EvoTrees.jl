# compute the gradient and hessian given target and predict
# linear
function update_grads!(loss::Linear, α::T, pred::AbstractMatrix{T}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    for i in eachindex(δ)
        δ[i] = 2 * (pred[i] - target[i]) * 𝑤[i]
        δ²[i] = 2 * 𝑤[i]
    end
end

# compute the gradient and hessian given target and predict
# logistic - on linear predictor
function update_grads!(loss::Logistic, α::T, pred::AbstractMatrix{T}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    for i in eachindex(δ)
        δ[i] = (sigmoid(pred[i]) * (1 - target[i]) - (1 - sigmoid(pred[i])) * target[i]) * 𝑤[i]
        δ²[i] = sigmoid(pred[i]) * (1 - sigmoid(pred[i])) * 𝑤[i]
    end
end

# compute the gradient and hessian given target and predict
# poisson
# Reference: https://isaacchanghau.github.io/post/loss_functions/
function update_grads!(loss::Poisson, α::T, pred::AbstractMatrix{T}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    for i in eachindex(δ)
        δ[i] = (exp(pred[i]) - target[i]) * 𝑤[i]
        δ²[i] = exp(pred[i]) * 𝑤[i]
    end
end

# compute the gradient and hessian given target and predict
# L1
function update_grads!(loss::L1, α::T, pred::AbstractMatrix{T}, target::AbstractArray{T, 1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    for i in eachindex(δ)
        δ[i] =  (α * max(target[i] - pred[i], 0) - (1-α) * max(pred[i] - target[i], 0)) * 𝑤[i]
    end
end

# compute the gradient and hessian given target and predict
# poisson
# Reference: https://isaacchanghau.github.io/post/loss_functions/
function update_grads!(loss::Softmax, α::T, pred::AbstractMatrix{T}, target::AbstractVector{Int}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat, L, M}
    # max = maximum(pred, dims=2)
    pred = pred .- maximum(pred, dims=2)
    sums = sum(exp.(pred), dims=2)
    for i in 1:size(pred,1)
        for j in 1:size(pred,2)
            if target[i] == j
                δ[i,j] = (exp(pred[i,j]) / sums[i] - 1) * 𝑤[i]
                δ²[i,j] =  1 / sums[i] * (1 - exp(pred[i,j]) / sums[i]) * 𝑤[i]
            else
                δ[i,j] = exp(pred[i,j]) / sums[i] * 𝑤[i]
                δ²[i,j] =  1 / sums[i] * (1 - exp(pred[i,j]) / sums[i]) * 𝑤[i]
            end
        end
    end
end

# compute the gradient and hessian given target and predict
# Quantile
function quantile_grads(pred, target, α)
    if target > pred; α
    elseif target < pred; α - 1
    end
end
function update_grads!(loss::Quantile, α::T, pred::AbstractMatrix{T}, target::AbstractVector{T}, δ::AbstractMatrix{T}, δ²::AbstractMatrix{T}, 𝑤::AbstractVector{T}) where T <: AbstractFloat
    @. δ =  quantile_grads(pred, target, α) * 𝑤
    @. δ² =  (target - pred) # No weighting on δ² as it would be applied on the quantile calculation
end

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
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GradientRegression, T <: AbstractFloat, L}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end

# Calculate the gain for a given split - MultiClassRegression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: MultiClassRegression, T <: AbstractFloat, L}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end

# Calculate the gain for a given split - L1Regression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: L1Regression, T <: AbstractFloat, L}
    gain = sum(abs.(∑δ))
    return gain
end

# Calculate the gain for a given split - QuantileRegression
function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: QuantileRegression, T <: AbstractFloat, L}
    # gain = (∑δ ^ 2 / (λ * ∑𝑤)) / 2
    gain = abs(∑δ) / (1 + λ)
    return gain
end
