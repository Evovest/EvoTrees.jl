using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Random: seed!

function hist_kernel!(h∇::CuDeviceArray, ∇::CuDeviceMatrix, x_bin::CuDeviceMatrix, is::CuDeviceVector, js::CuDeviceVector)
    k, tiy, tiz = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy = blockDim().z, blockDim().y
    bix, biy = blockIdx().x, blockIdx().y
    gdx = gridDim().x

    j = tiy + bdy * (biy - 1)
    if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        @inbounds for iter = 1:niter
            i = tiz + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            if i <= i_max
                @inbounds idx = is[i]
                @inbounds bin = x_bin[idx, jdx]
                hid = Base._to_linear_index(h∇, k, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu!(h∇, ∇, x_bin, is, js)
    # kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    # config = launch_configuration(kernel.fun)
    # max_threads = config.threads ÷ 4
    # max_blocks = config.blocks * 4
    max_threads = 256
    max_blocks = 16
    tx = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, tx)))
    tz = max(1, min(length(is), fld(max_threads, tx * ty)))
    threads = (tx, ty, tz)
    by = cld(length(js), ty)
    bz = min(cld(max_blocks, by), cld(length(is), tz))
    blocks = (bz, by, 1)
    h∇ .= 0
    @info "threads blocks" threads blocks
    @cuda threads=threads blocks=blocks hist_kernel!(h∇, ∇, x_bin, is, js)
    CUDA.synchronize()
    return nothing
end

seed!(123)
nbins = 32
nfeats = 2
nobs = Int(1e5)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
rowsample = 0.5
is = 1:nobs
js = 1:nfeats

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

CUDA.allowscalar(false)
update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
