using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools

# base kernel
function kernel_s4!(h::CuDeviceArray{T,3}, ∇::CuDeviceMatrix{T}, x_bin::CuDeviceMatrix{S}) where {T,S}

    nbins = size(h, 2)
    it, jt, kt = threadIdx().x, threadIdx().y, threadIdx().z
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    ig, jg = gridDim().x, gridDim().y
    j = jt + (jb - 1) * jd

    shared = @cuDynamicSharedMem(T, 3 * nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = size(x_bin, 1)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= size(x_bin, 1) && j <= size(x_bin, 2)
            # depends on shared to be assigned to a single feature
            k = 3 * (x_bin[i, j] - 1)
            @inbounds CUDA.atomic_add!(pointer(shared, k + kt), ∇[i, kt])
            # @inbounds CUDA.atomic_add!(pointer(shared, k + 1), ∇[i, 1])
            # @inbounds CUDA.atomic_add!(pointer(shared, k + 2), ∇[i, 2])
            # @inbounds CUDA.atomic_add!(pointer(shared, k + 3), ∇[i, 3])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for iter in 1:(nbins-1)÷id+1
        bin_id = it + id * (iter - 1)
        if bin_id <= nbins
            @inbounds k = Base._to_linear_index(h, 1, bin_id, j) - 1
            @inbounds CUDA.atomic_add!(pointer(h, k + kt), shared[3*(bin_id-1)+kt])
            # @inbounds CUDA.atomic_add!(pointer(h, k), shared[3*(bin_id-1)+1])
            # @inbounds CUDA.atomic_add!(pointer(h, k + 1), shared[3*(bin_id-1)+2])
            # @inbounds CUDA.atomic_add!(pointer(h, k + 2), shared[3*(bin_id-1)+3])
        end
    end
    sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu_s4!(h::AbstractArray{T,3}, ∇::AbstractMatrix{T}, x_bin::AbstractMatrix{S}; MAX_THREADS=256) where {T,S}
    thread_i = min(MAX_THREADS, size(x_bin, 1))
    thread_j = 1
    thread_k = 3
    threads = (thread_i, thread_j, thread_k)
    blocks = ceil.(Int, (16, size(x_bin, 2)))
    fill!(h, 0)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) * 3 kernel_s4!(h, ∇, x_bin)
    CUDA.synchronize()
    return
end

nbins = 64
nfeats = 100
nobs = Int(1e6)
h = [zeros(Float32, 3, nbins) for feat in 1nfeats];
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇_cpu = rand(Float32, 3, nobs);
h∇_cpu = zeros(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

∇_gpu = CuArray(∇_cpu)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇_cpu)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time hist_gpu_s4!(h∇_gpu, ∇_gpu, x_bin_gpu)
CUDA.@time hist_gpu_s4!(h∇_gpu, ∇_gpu, x_bin_gpu)
# desktop | 1K: 41.102 μs (24 allocations: 1.66 KiB)
# desktop | 10K: 59.142 μs (109 allocations: 9.09 KiB)
# desktop | 100K: 251.850 μs (109 allocations: 9.09 KiB)
# desktop | 1M: 2.203 ms (23 allocations: 1.33 KiB)
# desktop | 10M: 25.557 ms (110 allocations: 9.11 KiB)
@btime hist_gpu_s4!(h∇_gpu, ∇_gpu, x_bin_gpu)
