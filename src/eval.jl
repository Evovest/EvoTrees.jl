function eval_metric(::Val{:mse}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (p[1, i] - y[i])^2
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:rmse}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (p[1, i] - y[i])^2
    end
    eval = sqrt(eval / sum(w))
    return eval
end

function eval_metric(::Val{:mae}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * abs(p[1, i] - y[i])
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:logloss}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        pred = sigmoid(p[1, i])
        eval -= w[i] * (y[i] * log(pred) + (1 - y[i]) * log(1 - pred))
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:mlogloss}, p::AbstractMatrix{T}, y::AbstractVector{S}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat,S<:Integer}
    eval = zero(T)
    p_prob = exp.(p) ./ sum(exp.(p), dims = 1)
    @inbounds for i in eachindex(y)
        eval -= w[i] * log(p_prob[y[i], i])
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:poisson}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (exp(p[1, i]) * (1 - y[i]) + loggamma(y[i] + 1))
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:gaussian}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval += w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    eval /= sum(w)
    return eval
end

function eval_metric(::Val{:quantile}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    eval = zero(T)
    for i in eachindex(y)
        eval += w[i] * (α * max(y[i] - p[1, i], zero(T)) + (1 - α) * max(p[1, i] - y[i], zero(T)))
    end
    eval /= sum(w)
    return eval
end


function gini_raw(y::T, p::S) where {T,S}
    if length(y) < 2
        return 0.0
    end
    random = (1:length(p)) ./ length(p)
    l_sort = y[sortperm(p)]
    l_cum_w = cumsum(l_sort) ./ sum(y)
    gini = sum(l_cum_w .- random)
    return gini
end

function gini_norm(y::T, p::S) where {T,S}
    if length(y) < 2
        return 0.0
    end
    return gini_raw(y, p) / gini_raw(y, y)
end

function eval_metric(::Val{:gini}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α = 0.0) where {T<:AbstractFloat}
    return -gini_norm(y, view(p, 1, :))
end