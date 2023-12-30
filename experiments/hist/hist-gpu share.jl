using Revise
using CUDA
using StaticArrays
using StatsBase: sample
using BenchmarkTools

function hist_cpu_1!(hist, δ, idx)
    Threads.@threads for j in 1:size(idx, 2)
        for i in 1:size(idx, 1)
            @inbounds hist[idx[i,j], j] += δ[i,1]
        end
    end
    return
end

function hist_cpu_2!(h1::Matrix{T}, h2::Matrix{T}, hw::Matrix{T}, 
        δ1::Vector{T}, δ2::Vector{T}, w::Vector{T}, idx::Matrix{UInt8}) where {T}
    Threads.@threads for j in 1:size(idx, 2)
        @inbounds for i in 1:size(idx, 1)
            @inbounds h1[idx[i,j], j] += δ1[i]
            @inbounds h2[idx[i,j], j] += δ2[i]
            @inbounds hw[idx[i,j], j] += w[i]
        end
    end
    return
end


function hist_cpu_3!(h1::Matrix{T}, h2::Matrix{T}, hw::Matrix{T}, 
    δ1::Vector{T}, δ2::Vector{T}, 𝑤::Vector{T}, idx::Matrix{UInt8}, 𝑖, 𝑗) where {T}
    
    @inbounds Threads.@threads for j in 𝑗
        @inbounds for i in 𝑖
            h1[idx[i,j], j] += δ1[i]
            h2[idx[i,j], j] += δ2[i]
            hw[idx[i,j], j] += 𝑤[i]
        end
    end
    return
end

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
    for iter in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (iter - 1)
        if bin_id <= nbins
            @inbounds k = Base._to_linear_index(h, 1, bin_id, j)
            @inbounds CUDA.atomic_add!(pointer(h, k), shared[3 * (bin_id - 1) + 1])
            @inbounds CUDA.atomic_add!(pointer(h, k + 1), shared[3 * (bin_id - 1) + 2])
            @inbounds CUDA.atomic_add!(pointer(h, k + 2), shared[3 * (bin_id - 1) + 3])
        end
    end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s4!(h::AbstractArray{T,3}, x::AbstractMatrix{T}, id::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (16, size(id, 2)))
    fill!(h, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) * 3 kernel_s4!(h, x, id)
    return
end

# function hist_cpu_2!(hist, δ, idx)
#     Threads.@threads for j in 1:size(idx, 2)
#         for i in 1:size(idx, 1)
#             bin = idx[i,j]
#             @inbounds hist[1, bin, j] += δ[i,1]
#             @inbounds hist[2, bin, j] += δ[i,2]
#             @inbounds hist[3, bin, j] += δ[i,3]
#         end
#     end
#     return
# end

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


# base kernel
function kernel_s5!(hδ1::CuDeviceArray{T,3}, hδ2::CuDeviceArray{T,3}, h𝑤::CuDeviceMatrix{T}, δ1::CuDeviceMatrix{T}, δ2::CuDeviceMatrix{T}, 𝑤::CuDeviceVector{T}, xid::CuDeviceMatrix{S}) where {T,S}
    
    nbins = size(h𝑤, 1)
    it, jt = threadIdx().x, threadIdx().y
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    ig, jg = gridDim().x, gridDim().y
    j = jt + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, 3 * nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = size(𝑤, 1)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= size(xid, 1) && j <= size(xid, 2)
            # depends on shared to be assigned to a single feature
            k = 3 * (xid[i, j] - 1)
            @inbounds CUDA.atomic_add!(pointer(shared, k + 1), δ1[i, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 2), δ2[i, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 3), 𝑤[i])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for iter in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (iter - 1)
        if bin_id <= nbins
            @inbounds k = Base._to_linear_index(hδ1, 1, bin_id, j)
            @inbounds CUDA.atomic_add!(pointer(hδ1, k), shared[3 * (bin_id - 1) + 1])
            @inbounds CUDA.atomic_add!(pointer(hδ2, k), shared[3 * (bin_id - 1) + 2])
            @inbounds CUDA.atomic_add!(pointer(h𝑤, k), shared[3 * (bin_id - 1) + 3])
        end
    end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s5!(hδ1::AbstractArray{T,3}, hδ2::AbstractArray{T,3}, h𝑤::AbstractMatrix{T}, 
        δ1::AbstractMatrix{T}, δ2::AbstractMatrix{T}, 𝑤::AbstractVector{T}, 
        id::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (16, size(id, 2)))
    fill!(hδ1, 0)
    fill!(hδ2, 0)
    fill!(h𝑤, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h𝑤, 1) * 3 kernel_s5!(hδ1, hδ2, h𝑤, δ1, δ2, 𝑤, id)
    return
end

nbins = 32
ncol = 100
items = Int32(1e6)

δ1 = CUDA.rand(Float32, items, 1)
δ2 = CUDA.rand(Float32, items, 1)
𝑤 = CUDA.rand(Float32, items)
idx = UInt8.(rand(1:nbins, items, ncol))

hδ1 = CUDA.zeros(1, nbins, ncol)
hδ2 = CUDA.zeros(1, nbins, ncol)
h𝑤 =  CUDA.zeros(nbins, ncol)
idx_gpu = CuArray(idx)

@CUDA.time hist_gpu_s5!(hδ1, hδ2, h𝑤, δ1, δ2, 𝑤, idx_gpu, MAX_THREADS=128)
@btime CUDA.@sync hist_gpu_s5!($hδ1, $hδ2, $h𝑤, $δ1, $δ2, $𝑤, $idx_gpu, MAX_THREADS=128)


# base kernel
function kernel_s6!(hδ1::CuDeviceArray{T,3}, hδ2::CuDeviceArray{T,3}, h𝑤::CuDeviceMatrix{T}, δ1::CuDeviceMatrix{T}, δ2::CuDeviceMatrix{T}, 𝑤::CuDeviceVector{T}, xid::CuDeviceMatrix{S}, 𝑖, 𝑗) where {T,S}
    
    nbins = size(h𝑤, 1)
    it, jt = threadIdx().x, threadIdx().y
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    ig, jg = gridDim().x, gridDim().y
    j = jt + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, 3 * nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = size(𝑤, 1)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= length(𝑖) && j <= length(𝑗)
            # depends on shared to be assigned to a single feature
            i_idx = 𝑖[i]
            k = 3 * (xid[i_idx, 𝑗[j]] - 1)
            @inbounds CUDA.atomic_add!(pointer(shared, k + 1), δ1[i_idx, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 2), δ2[i_idx, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 3), 𝑤[i_idx])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for iter in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (iter - 1)
        if bin_id <= nbins
            @inbounds k = Base._to_linear_index(hδ1, 1, bin_id, 𝑗[j])
            @inbounds CUDA.atomic_add!(pointer(hδ1, k), shared[3 * (bin_id - 1) + 1])
            @inbounds CUDA.atomic_add!(pointer(hδ2, k), shared[3 * (bin_id - 1) + 2])
            @inbounds CUDA.atomic_add!(pointer(h𝑤, k), shared[3 * (bin_id - 1) + 3])
        end
    end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s6!(hδ1::AbstractArray{T,3}, hδ2::AbstractArray{T,3}, h𝑤::AbstractMatrix{T}, 
        δ1::AbstractMatrix{T}, δ2::AbstractMatrix{T}, 𝑤::AbstractVector{T}, 
        id::AbstractMatrix{S}, 𝑖, 𝑗; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (16, size(id, 2)))
    fill!(hδ1, 0)
    fill!(hδ2, 0)
    fill!(h𝑤, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h𝑤, 1) * 3 kernel_s6!(hδ1, hδ2, h𝑤, δ1, δ2, 𝑤, id, 𝑖, 𝑗)
    return
end

nbins = 32
ncol = 100
items = Int32(1e6)

δ1 = CUDA.rand(Float32, items, 1)
δ2 = CUDA.rand(Float32, items, 1)
𝑤 = CUDA.rand(Float32, items)
idx = UInt8.(rand(1:nbins, items, ncol))

hδ1 = CUDA.zeros(1, nbins, ncol)
hδ2 = CUDA.zeros(1, nbins, ncol)
h𝑤 =  CUDA.zeros(nbins, ncol)
idx_gpu = CuArray(idx)

𝑖 = CuArray(UInt32.(1:items))
𝑗 = CuArray(UInt32.(1:ncol))

hδ1_cpu = reshape(Array(hδ1), size(hδ1)[2:3])
hδ2_cpu = reshape(Array(hδ2), size(hδ2)[2:3])
h𝑤_cpu = Array(h𝑤)
δ1_cpu = reshape(Array(δ1), :)
δ2_cpu = reshape(Array(δ2), :)
𝑤_cpu = Array(𝑤)
𝑖_cpu = Array(𝑖)
𝑗_cpu = Array(𝑗)
idx_cpu = Array(idx_gpu)

@time hist_cpu_3!(hδ1_cpu, hδ2_cpu, h𝑤_cpu, δ1_cpu, δ2_cpu, 𝑤_cpu, idx_cpu, 𝑖_cpu, 𝑗_cpu)
@btime hist_cpu_3!(hδ1_cpu, hδ2_cpu, h𝑤_cpu, δ1_cpu, δ2_cpu, 𝑤_cpu, idx_cpu, 𝑖_cpu, 𝑗_cpu)

@CUDA.time hist_gpu_s6!(hδ1, hδ2, h𝑤, δ1, δ2, 𝑤, idx_gpu, 𝑖, 𝑗, MAX_THREADS=128)
@btime CUDA.@sync hist_gpu_s6!($hδ1, $hδ2, $h𝑤, $δ1, $δ2, $𝑤, $idx_gpu, $𝑖, $𝑗, MAX_THREADS=128)



function update_hist_cpu!(hist_δ::Matrix{SVector{L,T}}, hist_δ²::Matrix{SVector{L,T}}, hist_𝑤::Matrix{SVector{1,T}},
    δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}},
    X_bin, 𝑖, 𝑗) where {L,T,S}

    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    @inbounds Threads.@threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
            hist_δ²[X_bin[i,j], j] += δ²[i]
            hist_𝑤[X_bin[i,j], j] += 𝑤[i]
        end
    end
end

hδ1_cpu = SVector.(hδ1_cpu)
hδ2_cpu = SVector.(hδ2_cpu)
h𝑤_cpu = SVector.(h𝑤_cpu)
δ1_cpu = SVector.(δ1_cpu)
δ2_cpu = SVector.(δ2_cpu)
𝑤_cpu = SVector.(𝑤_cpu)
# 𝑖_cpu = Array(𝑖)
# 𝑗_cpu = Array(𝑗)
# idx_cpu = Array(idx_gpu)

@time update_hist_cpu!(hδ1_cpu, hδ2_cpu, h𝑤_cpu, δ1_cpu, δ2_cpu, 𝑤_cpu, idx_cpu, 𝑖_cpu, 𝑗_cpu)
@btime update_hist_cpu!(hδ1_cpu, hδ2_cpu, h𝑤_cpu, δ1_cpu, δ2_cpu, 𝑤_cpu, idx_cpu, 𝑖_cpu, 𝑗_cpu)
@btime update_hist_cpu!($hδ1_cpu, $hδ2_cpu, $h𝑤_cpu, $δ1_cpu, $δ2_cpu, $𝑤_cpu, $idx_cpu, $𝑖_cpu, $𝑗_cpu)


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