function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = mean((pred .- Y) .^ 2)
    return eval
end

function eval_metric(::Val{:rmse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = sqrt(mean((pred .- Y) .^ 2))
    return eval
end

function eval_metric(::Val{:mae}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = mean(abs.(pred .- Y))
    return eval
end

function eval_metric(::Val{:logloss}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = -mean(Y .* log.(max.(1e-8, sigmoid.(pred))) .+ (1 .- Y) .* log.(max.(1e-8, 1 .- sigmoid.(pred))))
    return eval
end

function eval_metric(::Val{:mlogloss}, pred::AbstractMatrix{T}, Y::AbstractVector{S}, α=0.0) where {T <: AbstractFloat, S <: Int}
    eval = 0.0
    for i in 1:length(Y)
        soft_pred = softmax(pred[i,:])
        eval -= log(soft_pred[Y[i]])
    end
    eval /= length(Y)
    return eval
end

function eval_metric(::Val{:quantile}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = mean(α .* max.(Y .- pred, 0) .+ (1-α) * max.(pred .- Y, 0))
    return eval
end
