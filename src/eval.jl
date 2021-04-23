# function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = mean((pred .- Y) .^ 2)
#     return eval
# end
function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval += (pred[1,i] - y[i])^2
    end
    eval /= length(y)
    return eval
end

function eval_metric(::Val{:rmse}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval += (pred[1,i] - y[i])^2
    end
    eval = sqrt(eval / length(y))
    return eval
end

function eval_metric(::Val{:mae}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval += abs(pred[1,i] - y[i])
    end
    eval /= length(y)
    return eval
end

function eval_metric(::Val{:logloss}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval -= y[i] * log(max(1e-8, pred[1,i])) + (1 - y[i]) * log(max(1e-8, 1 - pred[1,i]))
    end
    eval /= length(y)
    return eval
end

function eval_metric(::Val{:mlogloss}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where {T <: AbstractFloat,L,S <: Integer}
    eval = zero(T)
    # pred = pred - maximum.(pred)
    @inbounds for i in 1:length(y)
        pred[i] = pred[i] .- maximum(pred[i])
        soft_pred = exp.(pred[i]) / sum(exp.(pred[i]))
        eval -= log(soft_pred[y[i]])
    end
    eval /= length(y)
    return eval
end

function eval_metric(::Val{:poisson}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval += exp(pred[1,i]) * (1 - y[i]) + log(factorial(y[i]))
    end
    eval /= length(y)
    return eval
end

# gaussian
# pred[i][1] = μ
# pred[i][2] = log(σ)
# function eval_metric(::Val{:gaussian}, pred::Vector{SVector{L,T}}, Y::AbstractVector{T}, α=0.0) where {L, T <: AbstractFloat}
#     eval = zero(T)
#     @inbounds for i in 1:length(pred)
#         eval += pred[i][2] + (Y[i] - pred[i][1])^2 / (2*max(1e-8, exp(2*pred[i][2])))
#     end
#     eval /= length(Y)
#     return eval
# end

function eval_metric(::Val{:gaussian}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where {T <: AbstractFloat}
    eval = zero(T)
    @inbounds for i in 1:length(y)
        eval += pred[2,i] + (y[i] - pred[1,i])^2 / (2 * max(1e-8, exp(2 * pred[2,i])))
    end
    eval /= length(y)
    return eval
end

function eval_metric(::Val{:quantile}, pred::Vector{SVector{1,T}}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    for i in 1:length(y)
        eval += α * max(Y[i] - pred[1,i], zero(T)) + (1 - α) * max(pred[i][1] - y[i], zero(T))
    end
    eval /= length(y)
    return eval
end


function gini_raw(y::T, pred::S) where {T,S}
    if length(y) < 2 
        return 0.0
    end
    random = (1:length(pred)) ./ length(pred)
    l_sort = y[sortperm(pred)]
    l_cum_w = cumsum(l_sort) ./ sum(y)
    gini = sum(l_cum_w .- random)
    return gini
end

function gini_norm(y::T, pred::S) where {T,S}
    if length(y) < 2 
        return 0.0
    end
    return gini_raw(y, pred) / gini_raw(y, y)
end

function eval_metric(::Val{:gini}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    return -gini_norm(y, view(pred,1,:))
end