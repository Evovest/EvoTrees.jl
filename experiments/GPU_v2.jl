# using CUDA
using CUDA
# using Flux
# using GeometricFlux

nbins = 32
ncol = 100
items = Int(1e6)
hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items)
idx = rand(1:nbins, items, ncol)
𝑖 = collect(1:items)
𝑗 = collect(1:ncol)

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)
𝑖_gpu = CuArray(𝑖)
𝑗_gpu = CuArray(𝑗)

# CPU
function hist_cpu!(hist, δ, idx, 𝑖, 𝑗)
    Threads.@threads for j in 𝑗
        @inbounds for i in 𝑖
            hist[idx[i], j] += δ[i]
        end
    end
    return
end

function kernel_1!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_1!(h::CuMatrix{T}, x::CuVector{T}, id::CuMatrix{Int}, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_1!(h, x, id, 𝑖, 𝑗)
    return
end

@time hist_cpu!(hist, δ, idx)
CUDA.@time hist_gpu_1!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)




nbins = 32
ncol = 100
items = Int(2e6)
K = 1
hist = zeros(Float32, nbins, 3, ncol)
δ = rand(Float32, items, 3)
idx = rand(1:nbins, items, ncol)
𝑖 = collect(1:items)
𝑗 = collect(1:ncol)

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)
𝑖_gpu = CuArray(𝑖)
𝑗_gpu = CuArray(𝑗)

function kernel_2!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k1 = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 1, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k1), x[𝑖[i],1])
        @inbounds k2 = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 2, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k2), x[𝑖[i],2])
        @inbounds k3 = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 3, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k3), x[𝑖[i],3])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_2!(h::CuArray{T,3}, x::CuMatrix{T}, id::CuMatrix{Int}, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_2!(h, x, id, 𝑖, 𝑗)
    return
end

CUDA.@time hist_gpu_2!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)

hist_gpu_1 = Array(hist_gpu)
hist_gpu_2 = Array(hist_gpu)
diff1 = hist_gpu_2 - hist_gpu_1

######################################################################################################
# best approach: loop on K indicators
######################################################################################################
function kernel_3!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗, K) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        for k in 1:K
            @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], k, 𝑗[j])
            @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i],k])
        end
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_3!(h::CuArray{T,3}, x::CuMatrix{T}, id::CuMatrix{Int}, 𝑖, 𝑗, K; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads kernel_3!(h, x, id, 𝑖, 𝑗, K)
    return
end

hist_gpu_1 = Array(hist_gpu)
hist_gpu_2 = Array(hist_gpu)
diff2 = hist_gpu_2 - hist_gpu_1
diff2 - diff1

CUDA.@time hist_gpu_3!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, 3, MAX_THREADS=1024)



######################################################################################################
# 3D kernel - instead of iterating on K - Less efficient than the loop on Ks
######################################################################################################
function kernel_3D!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗, K) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], k, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i],k])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_3D!(h::CuArray{T,3}, x::CuMatrix{T}, id::CuMatrix{Int}, 𝑖, 𝑗, K; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_k = min(MAX_THREADS, K)
    thread_j = min(MAX_THREADS ÷ thread_k, length(𝑗))
    thread_i = min(MAX_THREADS ÷ (thread_k * thread_j), length(𝑖))
    threads = (thread_i, thread_j, thread_k)
    blocks = ceil.(Int, (length(𝑖), length(𝑗), K) ./ threads)
    @cuda blocks=blocks threads=threads kernel_3D!(h, x, id, 𝑖, 𝑗, K)
    return
end

CUDA.@time hist_gpu_3D!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, 3, MAX_THREADS=1024)

hist_gpu_1 = Array(hist_gpu)
hist_gpu_2 = Array(hist_gpu)
diff1 = hist_gpu_2 - hist_gpu_1


######################################################################################################
# 3D kernel - instead of iterating on K - No collision approach - single i thread - bad!
######################################################################################################
function kernel_3D2!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗, K) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if i <= length(𝑖) && j <= length(𝑗)
        # @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], k, 𝑗[j])
        @inbounds h[id[𝑖[i], 𝑗[j]], k, 𝑗[j]] += x[𝑖[i],k]
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_3D2!(h::CuArray{T,3}, x::CuMatrix{T}, id::CuMatrix{Int}, 𝑖, 𝑗, K; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_k = min(MAX_THREADS, K)
    thread_j = min(MAX_THREADS ÷ thread_k, length(𝑗))
    thread_i = 1
    threads = (thread_i, thread_j, thread_k)
    blocks = ceil.(Int, (length(𝑖), length(𝑗), K) ./ threads)
    @cuda blocks=blocks threads=threads kernel_3D2!(h, x, id, 𝑖, 𝑗, K)
    return
end

CUDA.@time hist_gpu_3D2!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, 3, MAX_THREADS=1024)

hist_gpu_1 = Array(hist_gpu)
hist_gpu_2 = Array(hist_gpu)
diff1 = hist_gpu_2 - hist_gpu_1
