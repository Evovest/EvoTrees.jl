using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools

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
    return nothing
end

function update_hist_gpu_vec!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel_vec!(view(h∇,:,:,1), ∇, view(x_bin, :, 1), is)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    @assert size(h∇, 1) <= max_threads "number of classes cannot be larger than 31 on GPU"
    ty = min(64, size(h∇, 1))
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    @sync for j in js
        @async kernel(view(h∇,:,:,j), ∇, view(x_bin, :, j), is; threads, blocks)
    end
    CUDA.synchronize()
    return nothing
end

nbins = 32
nfeats = 100
nobs = Int(1e6)
hist = zeros(Float32, nbins, nfeats);
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
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

@time update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu_vec!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)





"""
    hist_kernel!
"""
function hist_kernel_vec2!(h∇, ∇, x_bin, is)
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
    return nothing
end

function update_hist_gpu_vec2!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel_vec2!(h∇[js[1]], ∇, view(x_bin, :, js[1]), is)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    @assert size(h∇[1], 1) <= max_threads "number of classes cannot be larger than 31 on GPU"
    ty = min(64, size(h∇[1], 1))
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    @sync for j in js
        @async kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
    end
    CUDA.synchronize()
    return nothing
end

nbins = 32
nfeats = 100
nobs = Int(1e6)
hist = zeros(Float32, nbins, nfeats);
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = [zeros(Float32, 3, nbins) for n in 1:nfeats]
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

hist_gpu = CuArray(hist)
∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray.(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu_vec2!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu_vec2!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu_vec2!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)