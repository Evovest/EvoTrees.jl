using CUDAnative
using CuArrays
using StaticArrays
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
function kernel1!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= size(id, 1) && j <= size(id, 2)
        @inbounds k = Base._to_linear_index(h, id[i,j], j)
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu1!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}; MAX_THREADS=256) where {T<:AbstractFloat}
    # thread_i = min(MAX_THREADS, size(id, 1))
    # thread_j = min(MAX_THREADS ÷ thread_i, size(id, 2))
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(id, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(id, 2)) ./ threads)
    CuArrays.@sync begin
        @cuda blocks=blocks threads=threads kernel1!(h, x, id)
    end
    return
end

nbins = 32
ncol = 100
items = Int(2^20)
hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items, ncol)
idx = rand(1:nbins, items, ncol)

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

hist .- Array(hist_gpu)
sum(hist) - sum(Array(hist_gpu))

@CuArrays.time hist_gpu1!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=1024)
@time hist_cpu!(hist, δ, idx)
@btime hist_cpu!($hist, $δ, $idx)
@btime hist_gpu1!($hist_gpu, $δ_gpu, $idx_gpu, MAX_THREADS=1024)


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
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    # sync_threads()

    # if blockIdx().x == 1
    #     if tid <= nbins
    #         CUDAnative.atomic_add!(pointer(h,tid), shared[tid])
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
    CuArrays.@sync begin
        @cuda blocks=blocks threads=threads kernel2!(h, x, id, nbins)
    end
    return h
end

hist
@CuArrays.time hist_gpu2!(hist_gpu, δ_gpu, idx_gpu, 32, MAX_THREADS=1024)
@btime hist_gpu2!($hist_gpu, $δ_gpu, $idx_gpu, 32, MAX_THREADS=1024)
@device_code_warntype hist_gpu2!(hist_gpu, δ_gpu, idx_gpu, 32, MAX_THREADS=1024)
