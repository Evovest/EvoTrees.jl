using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using Random: seed!

"""
    hist_kernel!
"""
function hist_kernel_vec!(h∇, ∇, x_bin, is)
    tix, k = threadIdx().x, threadIdx().y
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(is)
    niter = cld(i_max, bdx * gdx)
    @inbounds for iter in 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            @inbounds idx = is[i]
            @inbounds bin = x_bin[idx]
            hid = Base._to_linear_index(h∇, k, bin)
            CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
        end
    end
    CUDA.sync_threads()
    return nothing
end

function update_hist_gpu_vec!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel_vec!(h∇[js[1]], ∇, view(x_bin, :, js[1]), is)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    ty = size(h∇[1], 1)
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    @sync for j in js
        @async h∇[j] .= 0
    end
    CUDA.synchronize()
    @sync for j in js
        @async kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
    end
    CUDA.synchronize()
    return nothing
end

seed!(123)
nbins = 32
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
h∇_gpu = CuArray.(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

CUDA.allowscalar(false)
@time update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@time CUDA.@sync update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@btime CUDA.@sync update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)

function update_hist_gpu_cpu!(h, h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel_vec!(h∇[js[1]], ∇, view(x_bin, :, js[1]), is)
    config = launch_configuration(kernel.fun)
    @info "config.threads" config.threads
    @info "config.blocks" config.blocks
    max_threads = config.threads
    max_blocks = config.blocks
    ty = size(h∇[1], 1)
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    CUDA.@sync for j in js
        @async h∇[j] .= 0
    end
    CUDA.synchronize()
    CUDA.@sync for j in js
        # @async kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
        kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
    end
    CUDA.synchronize()
    return nothing
end
function copy_gpu_cpu!(h, h∇, js)
    for j in js
        # @info "j" j
        copyto!(h[j], h∇[j])
    end
    CUDA.synchronize()
    return nothing
end
function combine!(h, h∇, ∇, x_bin, is, js)
    CUDA.@sync update_hist_gpu_cpu!(h, h∇, ∇, x_bin, is, js)
    CUDA.synchronize()
    CUDA.@sync copy_gpu_cpu!(h, h∇, js)
    CUDA.synchronize()
    return nothing
end
@time CUDA.@sync update_hist_gpu_cpu!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@time CUDA.@sync copy_gpu_cpu!(h∇, h∇_gpu, js)

@btime CUDA.@sync update_hist_gpu_cpu!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@btime CUDA.@sync copy_gpu_cpu!(h∇, h∇_gpu, js)

@time combine!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@time CUDA.@sync update_hist_gpu_cpu!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js)
@btime CUDA.@sync update_hist_gpu_cpu!($h∇, $h∇_gpu, $∇_gpu, $x_bin_gpu, $is_gpu, $js)
