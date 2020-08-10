using CUDA
# using Flux

items = Int(1e6)
δ = rand(Float32, items, 1)
δ² = rand(Float32, items, 1)
𝑤 = rand(Float32, items)
pred = rand(Float32, items, 1)
target = rand(Float32, items)

δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
𝑤_gpu = CuArray(𝑤)
pred_gpu = CuArray(pred)
target_gpu = CuArray(target)

function update_grads_gpu_linear_1!(pred::AbstractMatrix{T}, target::AbstractVector{T}, δ::AbstractMatrix{T}, δ²::AbstractMatrix{T}, 𝑤::AbstractVector{T}) where {T <: AbstractFloat}
    @. δ = 2f0 * (pred - target) * 𝑤
    @. δ² = 2f0 * 𝑤
    return
end


function kernel_linear_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ[i] = 2 * (p[i] - t[i]) * 𝑤[i]
    end
    return
end

function kernel_linear_δ²!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        @inbounds δ[i] = 2 * 𝑤[i]
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function grad_linear!(δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_linear_δ!(δ, p, t, 𝑤)
    @cuda blocks=blocks threads=threads kernel_linear_δ²!(δ², p, t, 𝑤)
    return
end

CUDA.@time update_grads_gpu_linear_1!(pred_gpu, target_gpu, δ_gpu, δ²_gpu, 𝑤_gpu)
CUDA.@time grad_linear!(δ_gpu, δ²_gpu, pred_gpu, target_gpu, 𝑤_gpu, MAX_THREADS=1024)

#################################################
# Gaussian
#################################################
items = Int(1e6)
δ = zeros(Float32, items, 1)
δ² = zeros(Float32, items, 1)
𝑤 = rand(Float32, items)
pred = rand(Float32, items, 1)
target = rand(Float32, items)

δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
𝑤_gpu = CuArray(𝑤)
pred_gpu = CuArray(pred)
target_gpu = CuArray(target)

function kernel_gauss_δ!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ[i,1] = (p[i,1] - t[i]) / max(Cfloat(1e-8), exp(2f0 * p[i,2])) * 𝑤[i]
        δ[i,2] = (1f0 - (p[i,1] - t[i])^2f0 / max(Cfloat(1e-8), exp(2f0 * p[i,2]))) * 𝑤[i]
    end
    return
end

function kernel_gauss_δ²!(δ::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, t::CuDeviceVector{T}, 𝑤::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(t)
        δ[i,1] = 𝑤[i] / max(Cfloat(1e-8), exp(2 * p[i,2]))
        δ[i,2] = 2 * 𝑤[i] / max(Cfloat(1e-8), exp(2 * pred[i,2])) * (p[i,1] - target[i])^2
    end
end

# base approach - block built along the cols first, the rows (limit collisions)
function grad_gaussian!(δ::CuMatrix{T}, δ²::CuMatrix{T}, p::CuMatrix{T}, t::CuVector{T}, 𝑤::CuVector{T}; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_i = min(MAX_THREADS, length(t))
    threads = (thread_i)
    blocks = ceil.(Int, (length(t)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_linear_δ!(δ, p, t, 𝑤)
    @cuda blocks=blocks threads=threads kernel_linear_δ²!(δ², p, t, 𝑤)
    return
end

CUDA.@time grad_gaussian!(δ_gpu, δ²_gpu, pred_gpu, target_gpu, 𝑤_gpu, MAX_THREADS=1024)
