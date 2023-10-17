using Revise
using CUDA
using StaticArrays
using StatsBase: sample
using BenchmarkTools

################################################
# TODO: test compact aggregation into log2 bins
# iter 5 times to cover 32 bins (8 for 256)
# - each block build histogram for many features -> (k, j)
# - 
################################################

function agg_share()
end

# base kernel
function kernel_share_1!(h::CuDeviceArray{T,3}, ∇, x_bin, is) where {T}
    
    nbins = size(h, 2)

    tix, tiy, k = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy = blockDim().x, blockDim().y
    bix, biy = blockIdx().x, blockIdx().y
    gdx, gdy = gridDim().x, gridDim().y
    
    j = tiy + (biy - 1) * bdy
    
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
function hist_share_1!(h::AbstractArray{T,3}, x::AbstractMatrix{T}, id::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (16, size(id, 2)))
    fill!(h, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) * 3 kernel_share_1!(h, ∇, x_bin, is)
    return
end

nbins = 64
nfeats = 100
nobs = Int32(1e6)
hist = zeros(Float32, 3, nbins, ncol)
∇ = rand(Float32, items, 3)
# idx = Int64.(rand(1:nbins, items, ncol))
idx = UInt8.(rand(1:nbins, items, ncol))

hist_gpu = CuArray(hist)
∇_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

@time hist_share_1!(hist, ∇, idx)
@btime hist_share_1!(hist, ∇, idx)
@CUDA.time hist_share_1!(hist_gpu, ∇_gpu, idx_gpu, MAX_THREADS=128)
@btime CUDA.@sync hist_share_1!($hist_gpu, $∇_gpu, $idx_gpu, MAX_THREADS=128)
