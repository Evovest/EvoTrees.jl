# function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = mean((pred .- Y) .^ 2)
#     return eval
# end
function eval_metric(::Val{:mse}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval += (pred[i][1] - Y[i]) ^ 2
    end
    eval /= length(pred)
    return eval
end

function eval_metric(::Val{:rmse}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval += (pred[i][1] - Y[i]) ^ 2
    end
    eval = sqrt(eval/length(pred))
    return eval
end

function eval_metric(::Val{:mae}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval += abs(pred[i][1] - Y[i])
    end
    eval /= length(pred)
    return eval
end

function eval_metric(::Val{:logloss}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval -= Y[i] * log(max(1e-8, sigmoid(pred[i][1]))) + (1 - Y[i]) * log(max(1e-8, 1 - sigmoid(pred[i][1])))
    end
    eval /= length(pred)
    return eval
end

function eval_metric(::Val{:mlogloss}, pred::Vector{SVector{1,T}}, Y::AbstractVector{S}, α=0.0) where {T <: AbstractFloat, S <: Int}
    eval = zero(T)
    for i in 1:length(Y)
        soft_pred = softmax(pred[i,:])
        eval -= log(soft_pred[Y[i]])
    end
    eval /= length(Y)
    return eval
end

function eval_metric(::Val{:quantile}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval += α .* max.(Y[i] - pred[i], 0) + (1-α) * max.(pred[i] - Y[i], 0)
    end
    eval /= length(pred)
    return eval
end
