using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using Random: seed!

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
    max_threads = config.threads ÷ 4
    max_blocks = config.blocks * 4
    k = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = min(64, max(1, min(length(is), fld(max_threads, k * ty))))
    threads = (k, ty, tx)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    synchronize()
    copyto!(h∇_cpu, h∇)
    Threads.@threads for j in jsc
        nbins = size(h[j], 2)
        @views h[j] .= h∇_cpu[:, 1:nbins, j]
    end
    return nothing
end

seed!(123)
nbins = 64
nfeats = 100
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = [zeros(Float32, 3, nbins) for n in 1:nfeats]
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_cpu = zeros(Float32, 3, nbins, nfeats)
h∇_gpu = CuArray(h∇_cpu)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

CUDA.allowscalar(false)
CUDA.@time update_hist_gpu!(h∇, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)

# laptop - 1M - without cpu copy: 5.426 ms (109 allocations: 6.64 KiB)
# laptop - 1M - with cpu copy: 5.458 ms (207 allocations: 17.08 KiB)
@btime update_hist_gpu!(h∇, h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
