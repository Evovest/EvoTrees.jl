# Gradient regression
function get_gain_gpu(::L, ∑δ::AbstractVector{T}, λ::T) where {L <: GradientRegression, T <: AbstractFloat}
    gain = ∑δ[1] ^ 2 / (∑δ[2] + λ * ∑δ[3]) / 2
    return gain
end

# Gaussian regression
function get_gain_gpu(::L, ∑δ::AbstractVector{T}, λ::T) where {L <: GaussianRegression, T <: AbstractFloat}
    gain = ∑δ[1] ^ 2 / (∑δ[2] + λ * ∑δ[3]) / 2 + ∑δ[4] ^ 2 / (∑δ[5] + λ * ∑δ[6]) / 2
    return gain
end

#####################
# linear
#####################
function kernel_linear_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds δ[i,1] = 2 * (p[i] - y[i]) * δ[i,3]
        @inbounds δ[i,2] =  2 * δ[i,3]
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
function kernel_logistic_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ[i] = (sigmoid(p[i]) * (1 - t[i]) - (1 - sigmoid(p[i])) * t[i]) * 𝑤[i]
    end
    return
end

function kernel_logistic_δ²!(δ²::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ²[i] = sigmoid(p[i]) * (1 - sigmoid(p[i])) * 𝑤[i]
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::Logistic, δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks = blocks threads = threads kernel_logistic_δ!(δ, p, t, 𝑤)
    @cuda blocks = blocks threads = threads kernel_logistic_δ²!(δ², p, t, 𝑤)
    return
end


################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ[i,1] = (p[i,1] - t[i]) / max(Cfloat(1e-5), exp(2f0 * p[i,2])) * 𝑤[i]
        δ[i,2] = (1f0 - (p[i,1] - t[i])^2 / max(Cfloat(1e-5), exp(2f0 * p[i,2]))) * 𝑤[i]
    end
    return
end

function kernel_gauss_δ²!(δ²::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ²[i,1] = 𝑤[i] / max(Cfloat(1e-5), exp(2 * p[i,2]))
        δ²[i,2] = 2 * 𝑤[i] / max(Cfloat(1e-5), exp(2 * p[i,2])) * (p[i,1] - t[i])^2
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::Gaussian, δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks = blocks threads = threads kernel_gauss_δ!(δ, p, t, 𝑤)
    @cuda blocks = blocks threads = threads kernel_gauss_δ²!(δ², p, t, 𝑤)
    return
end
