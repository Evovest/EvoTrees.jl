using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using StaticArrays

"""
    hist_kernel!
"""
function hist_kernel!(h∇::CuDeviceArray{T,3}, ∇::CuDeviceMatrix{S}, x_bin, is, js) where {T,S}
    tix, tiy = threadIdx().y, threadIdx().x,
    bdx, bdy = blockDim().y, blockDim().x
    bix, biy = blockIdx().y, blockIdx().x
    gdx = gridDim().z

    j = tiy + bdy * (biy - 1)
    if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        @inbounds for iter = 1:niter
            i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            if i <= i_max
                @inbounds idx = is[i]
                @inbounds bin = x_bin[idx, jdx]
                hid = Base._to_linear_index(h∇, k, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), T(∇[k, idx]))
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    k = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = max(1, min(length(is), fld(max_threads, k * ty)))
    threads = (k, ty, tx)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    return nothing
end

nbins = 32
nobs = Int(1e6)
nfeats = 100
rowsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(SVector{3,Float32}, nobs);
h∇ = zeros(SVector{3,Float64}, nbins, nfeats)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
CUDA.@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
# desktop | 1K: 46.332 μs (109 allocations: 9.09 KiB)
# desktop | 10K: 59.142 μs (109 allocations: 9.09 KiB)
# desktop | 100K: 251.850 μs (109 allocations: 9.09 KiB)
# desktop | 1M: 2.328 ms (110 allocations: 9.11 KiB)
# desktop | 10M: 25.557 ms (110 allocations: 9.11 KiB)
# @btime update_hist_gpu!(h, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)


"""
    minimal test
"""
function static_kernel!(a::CuDeviceVector, b::CuDeviceVector, c::CuDeviceVector)
    # function static_kernel!(a::CuDeviceArray, b::CuDeviceMatrix, c::CuDeviceMatrix)
    @inbounds for i in eachindex(a)
        a[i] = b[i] + c[i]
        # hid = Base._to_linear_index(h∇, k, bin, jdx)
        # CUDA.atomic_add!(pointer(h∇, hid), T(∇[k, idx]))
    end
    sync_threads()
    return nothing
end

function static_gpu!(a, b, c)
    kernel = @cuda launch = false static_kernel!(a, b, c)
    config = launch_configuration(kernel.fun)
    threads = (1,)
    blocks = (1,)
    kernel(a, b, c; threads, blocks)
    CUDA.synchronize()
    return nothing
end

a = zeros(SVector{3,Float64}, 10) |> cu
b = rand(SVector{3,Float64}, 10) |> cu
c = rand(SVector{3,Float64}, 10) |> cu
static_gpu!(a, b, c)

typeof(a) <: CuDeviceArray


"""
    minimal test - atomic - atomic not compatible with SVectors
"""
function static_kernel_atomic!(a::CuDeviceVector, b::CuDeviceVector, c::CuDeviceVector)
    # function static_kernel!(a::CuDeviceArray, b::CuDeviceMatrix, c::CuDeviceMatrix)
    i = threadIdx().x
    a[1] = b[i] + c[i]
    # CUDA.atomic_add!(pointer(a, i), b[i])
    sync_threads()
    return nothing
end

function static_gpu_atomic!(a, b, c)
    kernel = @cuda launch = false static_kernel_atomic!(a, b, c)
    config = launch_configuration(kernel.fun)
    threads = (length(a),)
    blocks = (1,)
    kernel(a, b, c; threads, blocks)
    CUDA.synchronize()
    return nothing
end

a = zeros(SVector{3,Float64}, 10) |> cu
b = rand(SVector{3,Float64}, 10) |> cu
c = rand(SVector{3,Float64}, 10) |> cu
static_gpu_atomic!(a, b, c)

b .+ c

typeof(a) <: CuDeviceArray


"""
    minimal test - atomic - atomic works with native types
"""
function static_kernel_atomic!(a::CuDeviceMatrix, b::CuDeviceMatrix, c::CuDeviceMatrix)
    # function static_kernel!(a::CuDeviceArray, b::CuDeviceMatrix, c::CuDeviceMatrix)
    i = threadIdx().x
    CUDA.atomic_add!(pointer(a, 1), b[i])
    sync_threads()
    return nothing
end

function static_gpu_atomic!(a, b, c)
    kernel = @cuda launch = false static_kernel_atomic!(a, b, c)
    config = launch_configuration(kernel.fun)
    threads = (length(a),)
    blocks = (1,)
    kernel(a, b, c; threads, blocks)
    CUDA.synchronize()
    return nothing
end

a = zeros(3, 10) |> cu
b = rand(3, 10) |> cu
c = rand(3, 10) |> cu
static_gpu_atomic!(a, b, c)

b .+ c
sum(b .+ c; dims=2)
sum(b .+ c; dims=1)

typeof(a) <: CuDeviceArray
