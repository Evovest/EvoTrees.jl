# function eval_metric(::Val{:mse}, pred::AbstractMatrix{T}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = mean((pred .- Y) .^ 2)
#     return eval
# end
function eval_metric(::Val{:mse}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α) where T <: AbstractFloat
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
    ϵ = T(2e-7)
    @inbounds for i in 1:length(pred)
        eval -= Y[i] * log(max(ϵ, sigmoid(pred[i][1]))) + (1 - Y[i]) * log(max(ϵ, 1 - sigmoid(pred[i][1])))
    end
    eval /= length(pred)
    return eval
end

function eval_metric(::Val{:mlogloss}, pred::Vector{SVector{L,T}}, Y::Vector{S}, α=0.0) where {T <: AbstractFloat, L, S <: Integer}
    eval = zero(T)
    ϵ = T(2e-7)
    @inbounds for i in 1:length(pred)
        # pred[i] = pred[i] .- maximum(pred[i])
        soft_pred = min.(1-ϵ, max.(ϵ, exp.(pred[i]) / sum(exp.(pred[i]))))
        p = pred[i] .- maximum(pred[i])
        soft_pred = min.(1-ϵ, max.(ϵ, exp.(p) / sum(exp.(p))))
        eval -= log(soft_pred[Y[i]])
    end
    eval /= length(Y)
    return eval
end

function eval_metric(::Val{:poisson}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    @inbounds for i in 1:length(pred)
        eval += exp(pred[i][1]) * (1 - Y[i]) + log(factorial(Y[i]))
    end
    eval /= length(pred)
    return eval
end

# gaussian
# pred[i][1] = μ
# pred[i][2] = log(σ)
function eval_metric(::Val{:gaussian}, pred::Vector{SVector{L,T}}, Y::AbstractVector{T}, α=0.0) where {L, T <: AbstractFloat}
    eval = zero(T)
    ϵ = T(2e-7)
    @inbounds for i in 1:length(pred)
        eval += pred[i][2] + (Y[i] - pred[i][1])^2 / (2*max(ϵ, exp(2*pred[i][2])))
    end
    eval /= length(Y)
    return eval
end

function eval_metric(::Val{:quantile}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    eval = zero(T)
    for i in 1:length(pred)
        eval += α * max(Y[i] - pred[i][1], zero(T)) + (1-α) * max(pred[i][1] - Y[i], zero(T))
    end
    eval /= length(pred)
    return eval
end


function gini_raw(labels::T, preds::S) where {T,S}
    if length(preds) < 2 
        return 0.0
    end
    random = (1:length(preds)) ./ length(preds)
    l_sort = labels[sortperm(preds)]
    l_cum_w = cumsum(l_sort) ./ sum(labels)
    gini = sum(l_cum_w .- random)
    return gini
end

function gini_norm(labels::T, preds::S) where {T,S}
    if length(labels) < 2 
        return 0.0
    end
    return gini_raw(labels, preds) / gini_raw(labels, labels)
end

function eval_metric(::Val{:gini}, pred::Vector{SVector{1,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    return -gini_norm(Y, pred)
end

function eval_metric(::Val{:gini}, pred::Vector{SVector{2,T}}, Y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
    return -gini_norm(Y, pred)
end