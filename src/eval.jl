function eval_metric(::Val{:mse}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}) where T <: AbstractFloat
    eval = mean(pred .- Y) .^ 2
    return eval
end

function eval_metric(::Val{:rmse}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}) where T <: AbstractFloat
    eval = sqrt(mean((pred .- Y) .^ 2))
    return eval
end

function eval_metric(::Val{:mae}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}) where T <: AbstractFloat
    eval = mean(abs.(pred .- Y))
    return eval
end

function eval_metric(::Val{:logloss}, pred::AbstractArray{T, 1}, Y::AbstractArray{T, 1}, tol=1e-15) where T <: AbstractFloat
    @. pred = max(pred, tol)
    @. pred = min(pred, 1-tol)
    eval = -mean(Y .* log.(pred) .+ (1 .- Y).*log.(1 .- pred))
    return eval
end
