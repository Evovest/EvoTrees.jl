function eval_metric(::Val{:mse}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, α=0.0) where T <: AbstractFloat
    eval = mean((pred .- Y) .^ 2)
    return eval
end

function eval_metric(::Val{:rmse}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, α=0.0) where T <: AbstractFloat
    eval = sqrt(mean((pred .- Y) .^ 2))
    return eval
end

function eval_metric(::Val{:mae}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, α=0.0) where T <: AbstractFloat
    eval = mean(abs.(pred .- Y))
    return eval
end

function eval_metric(::Val{:logloss}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, α=0.0) where T <: AbstractFloat
    eval = -mean(Y .* log.(max.(1e-8, sigmoid.(pred))) .+ (1 .- Y) .* log.(max.(1e-8, 1 .- sigmoid.(pred))))
    return eval
end

function eval_metric(::Val{:quantile}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, α=0.0) where T <: AbstractFloat
    eval = mean(α .* max.(Y .- pred, 0) .+ (1-α) * max.(pred .- Y, 0))
    return eval
end
