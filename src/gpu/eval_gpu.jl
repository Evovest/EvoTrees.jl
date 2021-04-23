# function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = mean((pred .- Y) .^ 2)
#     return eval
# end
# function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α) where T <: AbstractFloat
#     eval = zero(T)
#     @inbounds for i in 1:length(pred)
#         eval += (pred[i,1] - y[i]) ^ 2
#     end
#     eval /= length(pred)
#     return eval
# end

# function eval_metric(::Val{:rmse}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     @inbounds for i in 1:length(pred)
#         eval += (pred[i,1] - y[i]) ^ 2
#     end
#     eval = sqrt(eval/length(pred))
#     return eval
# end

# function eval_metric(::Val{:mae}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     @inbounds for i in 1:length(pred)
#         eval += abs(pred[i,1] - y[i])
#     end
#     eval /= length(pred)
#     return eval
# end

# function eval_metric(::Val{:logloss}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     @inbounds for i in 1:length(y)
#         eval -= y[i] * log(max(1e-8, sigmoid(pred[i,1]))) + (1 - y[i]) * log(max(1e-8, 1 - sigmoid(pred[i,1])))
#     end
#     eval /= length(y)
#     return eval
# end

# function eval_metric(::Val{:mlogloss}, pred::AbstractMatrix{T}, y::Vector{S}, α=0.0) where {T <: AbstractFloat, S <: Integer}
#     eval = zero(T)
#     L = length(y)
#     K = size(pred,2)
#     # pred = pred - maximum.(pred)
#     @inbounds for i in 1:L
#         pred[i] = pred[i] .- maximum(pred[i])
#         soft_pred = exp.(pred[i]) / sum(exp.(pred[i]))
#         eval -= log(soft_pred[y[i]])
#     end
#     eval /= length(y)
#     return eval
# end

# function eval_metric(::Val{:poisson}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     @inbounds for i in 1:length(y)
#         eval += exp(pred[i,1]) * (1 - y[i]) + log(factorial(y[i]))
#     end
#     eval /= length(y)
#     return eval
# end

# gaussian
# pred[i][1] = μ
# pred[i][2] = log(σ)
# function eval_metric(::Val{:gaussian}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where {L, T <: AbstractFloat}
#     eval = zero(T)
#     @inbounds for i in 1:length(y)
#         eval += pred[i,2] + (y[i] - pred[i,1])^2 / (2*max(1e-8, exp(2*pred[i,2])))
#     end
#     eval /= length(y)
#     return eval
# end

# function eval_metric(::Val{:quantile}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     for i in 1:length(y)
#         eval += α * max(y[i] - pred[i,1], zero(T)) + (1-α) * max(pred[i,1] - y[i], zero(T))
#     end
#     eval /= length(y)
#     return eval
# end
