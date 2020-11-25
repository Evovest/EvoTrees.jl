using Revise
using CUDA
using StaticArrays
using StatsBase: sample
using BenchmarkTools

function hist_cpu!(hist, δ, idx)
    Threads.@threads for j in 1:size(idx, 2)
        for i in 1:size(idx, 1)
            @inbounds hist[idx[i,j], j] += δ[i,1]
        end
    end
    return
end

# base kernel
function kernel_s3!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, xid::CuDeviceMatrix{S}) where {T,S}
    
    nbins = size(h, 1)
    it, jt = threadIdx().x, threadIdx().y
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    j = jt + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, nbins)
    fill!(h, 0)
    fill!(shared, 0)
    sync_threads()

    i_tot = size(x, 1)
    iter = 0
    while iter * id < i_tot
        i = it + id * iter
        if i <= size(xid, 1) && j <= size(xid, 2)
            # depends on shared to be assigned to a single feature
            @inbounds CUDA.atomic_add!(pointer(shared, xid[i, j]), x[i, 1])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for i in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (i - 1)
        if bin_id <= nbins
            # @inbounds h[bin_id, j] = shared[bin_id]
            @inbounds k = Base._to_linear_index(h, bin_id, j)
            @inbounds CUDA.atomic_add!(pointer(h, k), shared[bin_id])
        end
    end
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s3!(h::AbstractMatrix{T}, x::AbstractMatrix{T}, id::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (1, size(id, 2)))
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 1) kernel_s3!(h, x, id)
    return
end

nbins = 64
ncol = 100
items = Int32(1e6)
hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items, 1)
# idx = Int64.(rand(1:nbins, items, ncol))
idx = UInt8.(rand(1:nbins, items, ncol))

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

@time hist_cpu!(hist, δ, idx)
@btime hist_cpu!($hist, $δ, $idx)

@CUDA.time hist_gpu_s3!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=128)
@btime CUDA.@sync hist_gpu_s3!($hist_gpu, $δ_gpu, $idx_gpu, MAX_THREADS=128)



# base kernel
function kernel_s4!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, xid::CuDeviceMatrix{S}) where {T,S}
    
    nbins = size(h, 2)
    it, jt = threadIdx().x, threadIdx().y
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    ig, jg = gridDim().x, gridDim().y
    j = jt + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, 3 * nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = size(x, 1)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= size(xid, 1) && j <= size(xid, 2)
            # depends on shared to be assigned to a single feature
            k = 3 * (xid[i, j] - 1)
            @inbounds CUDA.atomic_add!(pointer(shared, k + 1), x[i, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 2), x[i, 2])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 3), x[i, 3])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    # for iter in 1:(nbins - 1) ÷ id + 1
    bin_id = it # + id * (iter - 1)
    if bin_id <= nbins
        @inbounds k = Base._to_linear_index(h, 1, bin_id, j)
        @inbounds CUDA.atomic_add!(pointer(h, k), shared[3 * (bin_id - 1) + 1])
        @inbounds CUDA.atomic_add!(pointer(h, k + 1), shared[3 * (bin_id - 1) + 2])
        @inbounds CUDA.atomic_add!(pointer(h, k + 2), shared[3 * (bin_id - 1) + 3])
    end
    # end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s4!(h::AbstractArray{T,3}, x::AbstractMatrix{T}, id::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (1, size(id, 2)))
    fill!(h, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) * 3 kernel_s4!(h, x, id)
    return
end

function hist_cpu_2!(hist, δ, idx)
    Threads.@threads for j in 1:size(idx, 2)
        for i in 1:size(idx, 1)
            bin = idx[i,j]
            @inbounds hist[1, bin, j] += δ[i,1]
            @inbounds hist[2, bin, j] += δ[i,2]
            @inbounds hist[3, bin, j] += δ[i,3]
        end
    end
    return
end

nbins = 64
ncol = 100
items = Int32(1e6)
hist = zeros(Float32, 3, nbins, ncol)
δ = rand(Float32, items, 3)
# idx = Int64.(rand(1:nbins, items, ncol))
idx = UInt8.(rand(1:nbins, items, ncol))

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

@time hist_cpu_2!(hist, δ, idx)
@btime hist_cpu_2!(hist, δ, idx)

@CUDA.time hist_gpu_s4!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=128)
@btime CUDA.@sync hist_gpu_s4!($hist_gpu, $δ_gpu, $idx_gpu, MAX_THREADS=128)


##############################################################
## Build histogram from a subsample idx
# base kernel
function kernel2!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k), x[𝑖[i], 𝑗[j]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu2!(h::CuMatrix{T}, x::CuMatrix{T}, id, 𝑖, 𝑗; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    # println("threads:", threads)
    # println("blocks:", blocks)
    CUDA.@sync begin
        @cuda blocks = blocks threads = threads kernel2!(h, x, id, 𝑖, 𝑗)
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
        @cuda blocks = blocks threads = threads kernel!(x, y)
    end
    return
end

x = rand(SVector{2,Float32}, Int(1e7))
y = rand(SVector{2,Float32}, Int(1e7))
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
        @cuda blocks = blocks threads = threads kernelS2!(h, x, id)
    end
    return
end

hist = zeros(SVector{2,Float32}, nbins, ncol)
δ = rand(SVector{2,Float32}, items, ncol)
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
        @cuda blocks = blocks threads = threads kernel3!(h, x, id, 𝑖, 𝑗)
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
function kernel2!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id, nbins) where {T <: AbstractFloat}
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
        @inbounds k = id[i,j] + nbins * (j - 1)
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
function hist_gpu2!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}, nbins; MAX_THREADS=256) where {T <: AbstractFloat}
    # thread_i = min(MAX_THREADS, size(id, 1))
    # thread_j = min(MAX_THREADS ÷ thread_i, size(id, 2))
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(id, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(id, 2)) ./ threads)
    CUDA.@sync begin
        @cuda blocks = blocks threads = threads kernel2!(h, x, id, nbins)
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
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, k), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(h::CuMatrix{T}, x::CuVector{T}, id, 𝑖, 𝑗; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks = blocks threads = threads kernel!(h, x, id, 𝑖, 𝑗)
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
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T <: AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(h::CuMatrix{T}, x::CuVector{T}, id::CuMatrix{UInt8}, 𝑖, 𝑗; MAX_THREADS=1024) where {T <: AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks = blocks threads = threads kernel!(h, x, id, 𝑖, 𝑗)
    return
end




using CUDA
using BenchmarkTools
N1 = Int(2^12)
x1 = rand(Float32, N1);
x2 = rand(Float32, N1);
x1g = CuArray(x1);
x2g = CuArray(x2);

function dot_atomic!(x, y, z)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(x)
        CUDA.atomic_add!(pointer(z, 1), x[idx] * y[idx])
    end
    return nothing
end

function bench_dot_atomic!(x, y, z, threads)
    numblocks = ceil(Int, N1 / threads)
    @cuda threads = threads blocks = numblocks dot_atomic!(x, y, z)
end

threads = 512
numblocks = ceil(Int, N1 / threads)
z0 = CUDA.zeros(1)
# @cuda threads=gthreads blocks=numblocks dot_atomic!(x1g, x2g, z0)
@btime CUDA.@sync bench_dot_atomic!($x1g, $x2g, $z0, threads)
#  17.323 ms (50 allocations: 1.67 KiB)

function dot_share!(x::CuDeviceVector{T}, y::CuDeviceVector{T}, z::CuDeviceVector{T}) where {T}

    tid = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    shared = CUDA.@cuStaticSharedMem(T, 64)
    fill!(shared, 0)
    sync_threads()

    if idx <= length(x)
        @inbounds shared[tid] = x[idx] * y[idx]
    end
    sync_threads()

    i = blockDim().x ÷ 2
    while i > 0
        if tid <= i
            @inbounds shared[tid] += shared[tid + i] # valid non atomic operation
            # CUDA.atomic_add!(pointer(shared, tid), shared[tid+i]) # invalid op - results in error
        end
        sync_threads()
        i ÷= 2
    end
    if tid == 1
        CUDA.atomic_add!(pointer(z, 1), shared[1])
    end
    return nothing
end

function bench_dot_share!(x, y, z, threads, numblocks)
    CUDA.@sync @cuda threads = threads blocks = numblocks dot_share!(x, y, z)
    return z
end

function wrap_share(x, y, threads)
    numblocks = ceil(Int, N1 / threads)
    z = CUDA.zeros(1)
    x = bench_dot_share!(x, y, z, threads, numblocks)
    return x
end

threads = 64
numblocks = ceil(Int, N1 / threads)
z = CUDA.zeros(1)
@cuda threads = threads blocks = numblocks dot_share!(x1g, x2g, z)
@time CUDA.@sync wrap_share(x1g, x2g, threads)
@btime CUDA.@sync wrap_share($x1g, $x2g, threads)
x = CUDA.@sync wrap_share(x1g, x2g, threads)
x1g' * x2g
@btime x1g' * x2g
@btime x1' * x2


function dot_share2!(x::CuDeviceVector{T}, z::CuDeviceVector{T}) where {T}

    tid = threadIdx().x
    bid = blockIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    shared = CUDA.@cuStaticSharedMem(T, 128)
    fill!(shared, 0f0)
    sync_threads()

    # if idx <= length(x)
    #     shared[tid] = x[idx] * y[idx]
    # end
    # sync_threads()

    i = blockDim().x ÷ 2
    while i > 0
        # if tid <= i
        if tid == 1
            # shared[tid] += shared[tid + i] # valid non atomic operation
            CUDA.atomic_add!(pointer(shared, tid), shared[tid + 1]) # invalid op - results in error
        end
        sync_threads()
        i ÷= 2
    end
    if tid == 1
        z[bid] += shared[1]
    end
    return nothing
end

function bench_dot_share2!(x, z, threads, numblocks)
    CUDA.@sync @cuda threads = threads blocks = numblocks dot_share2!(x, z)
    return sum(z)
end

function wrap_share2(x, threads)
    numblocks = ceil(Int, N1 / threads)
    z = CUDA.zeros(numblocks)
    # x = bench_dot_share2!(x, z, threads, numblocks)
    CUDA.@sync @cuda threads = threads blocks = numblocks dot_share2!(x, z)
    return(x)
end

threads = 128
numblocks = ceil(Int, N1 / threads)
z = CUDA.zeros(numblocks)
sum(z)
x1g' * x2g
CUDA.@sync @cuda threads = threads blocks = numblocks dot_share2!(x1g, x2g, z)
@btime CUDA.@sync wrap_share2($x1g, $x2g, threads)
x = CUDA.@sync wrap_share2(x1g, x2g, threads)

@btime x1g' * x2g


using CUDA
function kernel(x)
    shared = @cuStaticSharedMem(Float32, 2)
    fill!(shared, 1f0)
    sync_threads()
    # @atomic shared[threadIdx().x] += 0f0
    tid = threadIdx().x
    CUDA.atomic_add!(pointer(shared, tid), shared[tid + 1])
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(1)
@cuda kernel(x)
synchronize()


using CUDA
function kernel2(x, y)
    tid = threadIdx().x
    shared = @cuStaticSharedMem(Float32, 4)
    fill!(shared, 1f0)
    sync_threads()
    i = Int32(2)
    if i > 0
        CUDA.atomic_add!(pointer(shared, tid), shared[tid + 1])
        sync_threads()
        # i ÷= 2
    end
    sync_threads()
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(4)
y = CUDA.zeros(1)
@cuda threads = 2 kernel2(x, y)
synchronize()



using CUDA
function kernel1(x, y)
    tid = threadIdx().x
    shared = @cuStaticSharedMem(Float32, 4)
    fill!(shared, 1f0)
    sync_threads()
    i = Int32(2)
    if i > 0
        CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
        sync_threads()
        i ÷= 2
    end
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(4)
y = CUDA.zeros(1)
@cuda threads = 2 kernel1(x, y)
x
synchronize()



using CUDA
function kernel2(x, y)
    tid = threadIdx().x
    shared = @cuStaticSharedMem(Float32, 4)
    fill!(shared, 1f0)
    sync_threads()
    i = Int32(2)
    while i > 0
        CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
        sync_threads()
        i ÷= 2
    end
    sync_threads()
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(4)
y = CUDA.zeros(1)
@cuda threads = 2 kernel2(x, y)
x
synchronize()


using CUDA
function kernel3(x)
    tid = threadIdx().x
    shared = @cuStaticSharedMem(Float32, 4)
    fill!(shared, 1f0)
    sync_threads()
    CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
    sync_threads()
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(4)
@cuda threads = 2 kernel3(x)
x
synchronize()

using CUDA
function kernel4(x)
    tid = threadIdx().x
    shared = @cuStaticSharedMem(Float32, 4)
    fill!(shared, 1f0)
    sync_threads()
    CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
    sync_threads()
    CUDA.atomic_add!(pointer(shared, tid), shared[tid + 2])
    sync_threads()
    CUDA.atomic_add!(pointer(x, 1), shared[1])
    return
end

x = CUDA.zeros(4)
@cuda threads = 2 kernel4(x)
x
synchronize()