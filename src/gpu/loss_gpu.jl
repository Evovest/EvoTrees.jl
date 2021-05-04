# Gradient regression
function get_gain_gpu(::L, ∑::AbstractVector{T}, λ::T) where {L <: GradientRegression,T <: AbstractFloat}
    gain = ∑[1]^2 / (∑[2] + λ * ∑[3]) / 2
    return gain
end

# Gaussian regression
function get_gain_gpu(::L, ∑::AbstractVector{T}, λ::T) where {L <: GaussianRegression,T <: AbstractFloat}
    gain = ∑[1]^2 / (∑[3] + λ * ∑[5]) / 2 + ∑[2]^2 / (∑[4] + λ * ∑[5]) / 2
    return gain
end

#####################
# linear
#####################
function kernel_linear_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds δ[1,i] = 2 * (p[i] - y[i]) * δ[3,i]
        @inbounds δ[2,i] =  2 * δ[3,i]
    end
    return
end

function update_grads_gpu!(loss::Linear, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_linear_δ!(δ, p, y)
    CUDA.synchronize()
    return
end


#####################
# Logistic
#####################
function kernel_logistic_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds δ[1,i] = (p[i] * (1 - y[i]) - (1 - p[i]) * y[i]) * δ[3,i]
        @inbounds δ[2,i] = p[i] * (1 - p[i]) * δ[3,i]
    end
    return
end

function update_grads_gpu!(loss::Logistic, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_logistic_δ!(δ, p, y)
    CUDA.synchronize()
    return
end


################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(y)
        # first order gradients
        δ[1,i] = (p[1,i] - y[i]) / exp(2 * p[2,i]) *  δ[5,i]
        δ[2,i] = (1 - (p[1,i] - y[i])^2 / exp(2 * p[2,i])) *  δ[5,i]
        # second order gradients
        δ[3,i] =  δ[5,i] / exp(2 * p[2,i])
        δ[4,i] = 2 *  δ[5,i] / exp(2 * p[2,i]) * (p[1,i] - y[i])^2
    end
    return
end

function update_grads_gpu!(loss::Gaussian, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_gauss_δ!(δ, p, y)
    CUDA.synchronize()
    return
end


function update_childs_∑_gpu!(::L, nodes, n, bin, feat) where {L}
    nodes[n << 1].∑ .= nodes[n].hL[:, bin, feat]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[:, bin, feat]
    return nothing
end