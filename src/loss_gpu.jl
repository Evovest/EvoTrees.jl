#####################
# linear
#####################
function kernel_linear_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ[i] = 2 * (p[i] - t[i]) * 𝑤[i]
    end
    return
end

function kernel_linear_δ²!(δ²::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ²[i] = 2 * 𝑤[i]
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::S, δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {S <: GradientRegression, T<:AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_linear_δ!(δ, p, t, 𝑤)
    @cuda blocks=blocks threads=threads kernel_linear_δ²!(δ², p, t, 𝑤)
    return
end


# Gradient regression
function get_gain(loss::S, ∑δ::AbstractVector{T}, ∑δ²::AbstractVector{T}, ∑𝑤::T, λ::T) where {S <: GradientRegression, T <: AbstractFloat}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end


# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function kernel_gauss_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ[i,1] = (p[i,1] - t[i]) / max(Cfloat(1e-8), exp(2f0 * p[i,2])) * 𝑤[i]
        δ[i,2] = (1f0 - (p[i,1] - t[i])^2f0 / max(Cfloat(1e-8), exp(2f0 * p[i,2]))) * 𝑤[i]
    end
    return
end

function kernel_gauss_δ²!(δ²::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ²[i,1] = 𝑤[i] / max(Cfloat(1e-8), exp(2 * p[i,2]))
        δ²[i,2] = 2 * 𝑤[i] / max(Cfloat(1e-8), exp(2 * p[i,2])) * (p[i,1] - target[i])^2
    end
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_grads_gpu!(loss::S, δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {S <: GaussianRegression, T<:AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_linear_δ!(δ, p, t, 𝑤)
    @cuda blocks=blocks threads=threads kernel_linear_δ²!(δ², p, t, 𝑤)
    return
end

# GaussianRegression
function get_gain(loss::S, ∑δ::T, ∑δ²::T, ∑𝑤::T, λ::T) where {S <: GaussianRegression, T <: AbstractFloat}
    gain = sum((∑δ .^ 2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end
