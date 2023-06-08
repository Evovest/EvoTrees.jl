using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools

"""
    hist_kernel!
"""
function hist_kernel!(h∇, ∇, x_bin, is, js)
    tix, tiy = threadIdx().x, threadIdx().y
    bdx, bdy = blockDim().x, blockDim().y
    bix, biy = blockIdx().x, blockIdx().y
    gdx = gridDim().x

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
                hid = Base._to_linear_index(h∇, 1, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), ∇[1, idx])
                CUDA.atomic_add!(pointer(h∇, hid+1), ∇[2, idx])
                CUDA.atomic_add!(pointer(h∇, hid+2), ∇[3, idx])
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
    # @info "config.blocks" config.blocks
    max_threads = config.threads
    max_blocks = config.blocks
    ty = max(1, min(length(js), max_threads))
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (bx, by, 1)
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    return nothing
end

nbins = 32
nfeats = 100
nobs = Int(1e6)
hist = zeros(Float32, nbins, nfeats);
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = rand(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

hist_gpu = CuArray(hist)
∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
CUDA.@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)