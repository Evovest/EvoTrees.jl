# Gradient regression
function get_gain_gpu(::L, ∑δ::AbstractVector{T}, λ::T) where {L <: GradientRegression,T <: AbstractFloat}
    gain = ∑δ[1]^2 / (∑δ[2] + λ * ∑δ[3]) / 2
    return gain
end

# Gaussian regression
function get_gain_gpu(::L, ∑δ::AbstractVector{T}, λ::T) where {L <: GaussianRegression,T <: AbstractFloat}
    gain = ∑δ[1]^2 / (∑δ[3] + λ * ∑δ[5]) / 2 + ∑δ[2]^2 / (∑δ[4] + λ * ∑δ[5]) / 2
    return gain
end

#####################
# linear
#####################
function kernel_linear_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ϵ = Cfloat(2e-7)
    if i <= length(y)
        @inbounds δ[i,1] = 2 * (p[i] - y[i]) * δ[i,3]
        @inbounds δ[i,2] =  max(ϵ, 2 * δ[i,3])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::Linear, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_i = min(MAX_THREADS, length(y))
    threads = (thread_i)
    blocks = ceil.(Int, (length(y)) ./ threads)
    @cuda blocks = blocks threads = threads kernel_linear_δ!(δ, p, y)
    return
end


#####################
# Logistic
#####################
function kernel_logistic_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ϵ = Cfloat(2e-7)
    if i <= length(y)
        @inbounds δ[i,1] = (sigmoid(p[i]) * (1 - y[i]) - (1 - sigmoid(p[i])) * y[i]) * δ[i,3]
        @inbounds δ[i,2] = max(ϵ, sigmoid(p[i]) * (1 - sigmoid(p[i])) * δ[i,3])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::Logistic, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_i = min(MAX_THREADS, length(y))
    threads = (thread_i)
    blocks = ceil.(Int, (length(y)) ./ threads)
    @cuda blocks = blocks threads = threads kernel_logistic_δ!(δ, p, y)
    return
end


################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ϵ = Cfloat(2e-7)
    if i <= length(y)

        # first order gradients
        δ[i,1] = (p[i,1] - y[i]) / max(ϵ, exp(2f0 * p[i,2])) *  δ[i,5]
        δ[i,2] = (1f0 - (p[i,1] - y[i])^2 / max(ϵ, exp(2f0 * p[i,2]))) *  δ[i,5]

        # second order gradients
        δ[i,3] =  max(ϵ, δ[i,5] / max(ϵ, exp(2 * p[i,2])))
        δ[i,4] = max(ϵ, 2 * δ[i,5] / max(ϵ, exp(2 * p[i,2])) * (p[i,1] - y[i])^2)
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::Gaussian, δ::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_i = min(MAX_THREADS, length(y))
    threads = (thread_i)
    blocks = ceil.(Int, (length(y)) ./ threads)
    @cuda blocks = blocks threads = threads kernel_gauss_δ!(δ, p, y)
    return
end
