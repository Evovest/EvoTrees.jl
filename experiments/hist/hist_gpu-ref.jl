using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools

"""
    hist_kernel!
"""
function hist_kernel!(h∇::CuDeviceArray{T,3}, ∇::CuDeviceMatrix{S}, x_bin, is, js) where {T,S}
    tix, tiy, k = threadIdx().z, threadIdx().y, threadIdx().x
    bdx, bdy = blockDim().z, blockDim().y
    bix, biy = blockIdx().z, blockIdx().y
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

function update_hist_gpu!(h, h∇_cpu, h∇, ∇, x_bin, is, js, jsc)
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
    copyto!(h∇_cpu, h∇)
    Threads.@threads for j in jsc
        nbins = size(h[j], 2)
        @views h[j] .= h∇_cpu[:, 1:nbins, j]
    end
    return nothing
end

nbins = 64
nfeats = 100
nobs = Int(1e3)
h = [zeros(Float32, 3, nbins) for feat in 1:nfeats];
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇_cpu = rand(Float32, 3, nobs);
h∇_cpu = rand(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

hist_gpu = CuArray(hist)
∇_gpu = CuArray(∇_cpu)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇_cpu)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
CUDA.@time update_hist_gpu!(h, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
# desktop | 1K: 46.332 μs (109 allocations: 9.09 KiB)
# desktop | 10K: 59.142 μs (109 allocations: 9.09 KiB)
# desktop | 100K: 251.850 μs (109 allocations: 9.09 KiB)
# desktop | 1M: 2.328 ms (110 allocations: 9.11 KiB)
# desktop | 10M: 25.557 ms (110 allocations: 9.11 KiB)
@btime update_hist_gpu!(h, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
