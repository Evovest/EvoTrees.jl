using Revise
using CUDA
using StaticArrays
using StatsBase: sample
using BenchmarkTools

function hist_cpu!(hist, δ, idx)
    Threads.@threads for j in 1:size(idx,2)
        for i in 1:size(idx,1)
            @inbounds hist[idx[i,j], j] += δ[i,j]
        end
    end
    return
end

# base kernel
function kernel1!(h, x, id)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= size(id, 1) && j <= size(id, 2)
        @inbounds k = Base._to_linear_index(h, id[i,j], j)
        # @inbounds k = id[i,j] + 32 * (j-1)
        @inbounds CUDA.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu1!(h::AbstractMatrix{T}, x::AbstractMatrix{T}, id; MAX_THREADS=256) where {T}
    # thread_i = min(MAX_THREADS, size(id, 1))
    # thread_j = min(MAX_THREADS ÷ thread_i, size(id, 2))
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(id, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(id, 2)) .÷ threads)
    # println("threads:", threads)
    # println("blocks:", blocks)
    CUDA.@sync begin
        @cuda blocks=blocks threads=threads kernel1!(h, x, id)
    end
    return
end

nbins = 32
ncol = 100
items = Int32(2^20)
hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items, ncol)
idx = Int64.(rand(1:nbins, items, ncol))

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

hist .- Array(hist_gpu)
sum(hist) - sum(Array(hist_gpu))

@CUDA.time hist_gpu1!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=1024)
@time hist_cpu!(hist, δ, idx)
@btime hist_cpu!($hist, $δ, $idx)
@btime hist_gpu1!($hist_gpu, $δ_gpu, $idx_gpu, MAX_THREADS=1024)
# test on view
@CUDA.time hist_gpu1!(hist_gpu, view(δ_gpu, 1:items÷2, 1:ncol÷2), view(idx_gpu, 1:items÷2, 1:ncol÷2), MAX_THREADS=1024)

size(δ_gpu)
size(view(δ_gpu, 1:items÷2, 1:ncol÷2))

##############################################################
## Build histogram from a subsample idx
# base kernel
function kernel2!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k), x[𝑖[i], 𝑗[j]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu2!(h::CuMatrix{T}, x::CuMatrix{T}, id, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    # println("threads:", threads)
    # println("blocks:", blocks)
    CUDA.@sync begin
        @cuda blocks=blocks threads=threads kernel2!(h, x, id, 𝑖, 𝑗)
    end
    return
end

hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items, ncol)
idx = rand(1:nbins, items, ncol)
𝑖 = sample(1:items, items ÷ 2, replace=false, ordered=true)
𝑗 = sample(1:ncol, ncol ÷ 2, replace=false, ordered=true)
hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)
𝑖_gpu = CuArray(𝑖)
𝑗_gpu = CuArray(𝑗)

@CUDA.time hist_gpu2!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)
@btime hist_gpu2!($hist_gpu, $δ_gpu, $idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)


#############################################
# test for SVector - basic test - success!
function kernel!(x, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(x)
        # @inbounds x[i] += y[i]
        k = Base._to_linear_index(x, i)
        CUDA.atomic_add!(pointer(x, k), y[i])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(x, y; MAX_THREADS=1024)
    thread_i = min(MAX_THREADS, length(x))
    threads = (thread_i)
    blocks = ceil.(Int, length(x) .÷ threads)
    CUDA.@sync begin
        @cuda blocks=blocks threads=threads kernel!(x, y)
    end
    return
end

x = rand(SVector{2, Float32}, Int(1e7))
y = rand(SVector{2, Float32}, Int(1e7))
x = rand(Float32, Int(1e7))
y = rand(Float32, Int(1e7))

x_gpu = CuArray(x)
y_gpu = CuArray(y)

@CuArrays.time hist_gpu!(x_gpu, y_gpu)
@btime hist_gpu!($x_gpu, $y_gpu)


#############################################
# test for SVector - real test
function kernelS2!(h, x, id)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= size(id, 1) && j <= size(id, 2)
        @inbounds k = Base._to_linear_index(h, id[i,j], j)
        # @inbounds k = id[i,j] + 32 * (j-1)
        # @inbounds CUDAnative.atomic_add!(pointer(h, k), x[i,j])
        # h[id[i,j],j] += x[i,j]
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpuS2!(h, x, id; MAX_THREADS=256) where {T}
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(id, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(id, 2)) .÷ threads)
    println("threads:", threads)
    println("blocks:", blocks)
    CUDA.@sync begin
        @cuda blocks=blocks threads=threads kernelS2!(h, x, id)
    end
    return
end

hist = zeros(SVector{2, Float32}, nbins, ncol)
δ = rand(SVector{2, Float32}, items, ncol)
idx = rand(1:nbins, items, ncol)
hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

@CuArrays.time hist_gpuS2!(hist_gpu, δ_gpu, idx_gpu)
@btime hist_gpuS2!($hist_gpu, $δ_gpu, $idx_gpu)


##############################################################
## Build histogram from a subsample idx
# accumulate all gradient single pass
function kernel3!(h, x, id, 𝑖, 𝑗)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j], 1)
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[𝑖[i], 𝑗[j], 1])
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j], 2)
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[𝑖[i], 𝑗[j], 2])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu3!(h, x, id, 𝑖, 𝑗; MAX_THREADS=1024)
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    # println("threads:", threads)
    # println("blocks:", blocks)
    CuArrays.@sync begin
        @cuda blocks=blocks threads=threads kernel3!(h, x, id, 𝑖, 𝑗)
    end
    return
end

hist = zeros(Float32, nbins, ncol, 2)
δ = rand(Float32, items, ncol, 2)
idx = rand(1:nbins, items, ncol)
𝑖 = sample(1:items, items ÷ 2, replace=false, ordered=true)
𝑗 = sample(1:ncol, ncol ÷ 2, replace=false, ordered=true)
hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)
𝑖_gpu = CuArray(𝑖)
𝑗_gpu = CuArray(𝑗)

@CuArrays.time hist_gpu3!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)
@btime hist_gpu3!($hist_gpu, $δ_gpu, $idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)




# accumulate in shared memory histograms
function kernel2!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id, nbins) where {T<:AbstractFloat}
    tid = threadIdx().x
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # shared memory on block of length nbins
    # To Do: nbins cannot be passed as argument - dynamic shared memory generate kernel through macro
    # shared = CUDAnative.@cuStaticSharedMem(T, 32)
    # fill!(shared, 0)
    # sync_threads()

    # accumulate in per block histogram
    # Why is the atomic add on shared much longer than atomic on h in global mem?
    if i <= size(id, 1) && j <= size(id, 2)
        # should be the legit way to go - 70ms
        # @inbounds CUDAnative.atomic_add!(pointer(shared, id[i,j]), x[i,j])

        # unsafe (collisions) on shared mem: 3.0ms
        # @inbounds shared[id[i,j]] = x[i,j]

        # unsafe (collisions) add on global memory - 3.6ms
        # @inbounds h[id[i,j],j] += x[i,j]

        # atomic add on global hist - 3.2ms
        @inbounds k = id[i,j] + nbins * (j-1)
        @inbounds CUDA.atomic_add!(pointer(h, k), x[i,j])
    end
    # sync_threads()

    # if blockIdx().x == 1
    #     if tid <= nbins
    #         CUDA.atomic_add!(pointer(h,tid), shared[tid])
    #     end
    # end
    return
end

# shared memory -
function hist_gpu2!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}, nbins; MAX_THREADS=256) where {T<:AbstractFloat}
    # thread_i = min(MAX_THREADS, size(id, 1))
    # thread_j = min(MAX_THREADS ÷ thread_i, size(id, 2))
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(id, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(id, 2)) ./ threads)
    CUDA.@sync begin
        @cuda blocks=blocks threads=threads kernel2!(h, x, id, nbins)
    end
    return h
end

@CuArrays.time hist_gpu2!(hist_gpu, δ_gpu, idx_gpu, 32, MAX_THREADS=1024)
@btime hist_gpu2!($hist_gpu, $δ_gpu, $idx_gpu, 32, MAX_THREADS=1024)
@device_code_warntype hist_gpu2!(hist_gpu, δ_gpu, idx_gpu, 32, MAX_THREADS=1024)




######################################
# Appoach 1
######################################
# GPU - apply along the features axis
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(h::CuMatrix{T}, x::CuVector{T}, id, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads kernel!(h, x, id, 𝑖, 𝑗)
    return
end

hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items)
idx = rand(1:nbins, items, ncol)
𝑖 = sample(1:items, items ÷ 2, replace=false, ordered=true)
𝑗 = sample(1:ncol, ncol ÷ 2, replace=false, ordered=true)
hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)
𝑖_gpu = CuArray(𝑖)
𝑗_gpu = CuArray(𝑗)

@CUDA.time hist_gpu!(hist_gpu, δ_gpu, idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)
@btime hist_gpu!($hist_gpu, $δ_gpu, $idx_gpu, 𝑖_gpu, 𝑗_gpu, MAX_THREADS=1024)


######################################
# Appoach 2 - Loop for assigning command grad to appropriate bin per column
# Idea: exploit the fact that there's a single grad per row: take that grad and add it to each column bin
######################################
# GPU - apply along the features axis
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(h::CuMatrix{T}, x::CuVector{T}, id::CuMatrix{UInt8}, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads kernel!(h, x, id, 𝑖, 𝑗)
    return
end
