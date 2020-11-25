using Revise
using CUDA
using StaticArrays
using StatsBase: sample
using BenchmarkTools

nbins = 10
items = Int32(1e6)
hist = zeros(Float32, nbins)
x = ones(Float32, items)
idx = Int64.(rand(1:nbins, items))

hist_gpu = CuArray(hist)
x_gpu = CuArray(x)
idx_gpu = CuArray(idx)

hist .- Array(hist_gpu)
sum(hist) - sum(Array(hist_gpu))

# base kernel
function kernel!(h::CuDeviceVector{T}, x::CuDeviceVector{T}, xid::CuDeviceVector{S}) where {T,S}
    
    nbins = size(h, 1)
    it = threadIdx().x
    ib = blockIdx().x
    id = blockDim().x
    
    shared = @cuDynamicSharedMem(T, nbins)
    fill!(shared, 0)
    fill!(h, 0)
    sync_threads()

    i_tot = size(x, 1)
    iter = 0
    while iter * id < i_tot
        i = it + id * iter
        if i <= size(xid, 1)
            @inbounds k = Base._to_linear_index(h, xid[i])
            @inbounds CUDA.atomic_add!(pointer(shared, k), x[i])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for i in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (i - 1)
        if bin_id <= nbins
            @inbounds CUDA.atomic_add!(pointer(h, bin_id), shared[bin_id])
        end
    end
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist!(h::AbstractVector{T}, x::AbstractVector{T}, xid::AbstractVector{S}; MAX_THREADS=256) where {T,S}
    threads = min(MAX_THREADS, size(xid, 1))
    @cuda blocks = 1 threads = threads shmem = sizeof(T) * size(h,1) kernel!(h, x, xid)
    return
end

@btime CUDA.@sync hist!($hist_gpu, $x_gpu, $idx_gpu, MAX_THREADS=1024)

