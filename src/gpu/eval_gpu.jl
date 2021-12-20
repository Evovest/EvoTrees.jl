"""
    MSE
"""
function eval_mse_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = w[i] * (p[1, i] - y[i])^2
    end
    return nothing
end

function eval_metric(::Val{:mse}, eval::AbstractVector{T}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α; MAX_THREADS = 1024) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, length(y) / threads)
    @cuda blocks = blocks threads = threads eval_mse_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

"""
    Logloss
"""
function eval_logloss_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds eval[i] = w[i] * (-y[i] * log(pred) + (y[i] - 1) * log(1 - pred))
    end
    return nothing
end

function eval_metric(::Val{:logloss}, eval::AbstractVector{T}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α; MAX_THREADS = 1024) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, length(y) / threads)
    @cuda blocks = blocks threads = threads eval_logloss_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end


"""
    Gaussian
"""
function eval_gaussian_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return nothing
end

function eval_metric(::Val{:gaussian}, eval::AbstractVector{T}, p::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}, α; MAX_THREADS = 1024) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, length(y) / threads)
    @cuda blocks = blocks threads = threads eval_gaussian_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end


"""
    Poisson
        Unsupported factorial/gamma on CUDA at the moment
"""
# function eval_pois_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if i <= length(y)
#         @inbounds eval[i] = exp(p[1,i]) * (1 - y[i]) + loggamma(y[i] + 1)
#     end
#     return nothing
# end

# function eval_metric(::Val{:poisson}, eval::AbstractVector{T}, p::AbstractMatrix{T}, y::AbstractVector{T}, α; MAX_THREADS=1024) where T <: AbstractFloat
#     threads = min(MAX_THREADS, length(y))
#     blocks = ceil(Int, length(y) / threads)
#     @cuda blocks = blocks threads = threads eval_pois_kernel!(eval, p, y)
#     CUDA.synchronize()
#     return mean(eval)
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

# function eval_metric(::Val{:quantile}, pred::AbstractMatrix{T}, y::AbstractVector{T}, α=0.0) where T <: AbstractFloat
#     eval = zero(T)
#     for i in 1:length(y)
#         eval += α * max(y[i] - pred[i,1], zero(T)) + (1-α) * max(pred[i,1] - y[i], zero(T))
#     end
#     eval /= length(y)
#     return eval
# end
