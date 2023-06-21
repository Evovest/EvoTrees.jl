using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Random: seed!

"""
    hist_kernel!
"""
function hist_kernel!(h∇, ∇, x_bin, is, js)
    tix, tiy, k = threadIdx().x, threadIdx().y, threadIdx().z
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
                hid = Base._to_linear_index(h∇, k, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu2!(h, h∇, ∇, x_bin, is, js, jsc)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
    # @info "config.threads" config.threads
    # @info "config.blocks" config.blocks
    max_threads = config.threads ÷ 4
    max_blocks = config.blocks * 4
    @assert size(h∇, 1) <= max_threads "number of classes cannot be larger than 31 on GPU"
    tz = min(64, size(h∇, 1))
    ty = max(1, min(length(js), fld(max_threads, tz)))
    tx = max(1, min(length(is), fld(max_threads, tz * ty)))
    threads = (tx, ty, tz)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (bx, by, 1)
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    # @inbounds for j in jsc
    #     copyto!(h[j], view(h∇, :, :, j))
    # end
    @inbounds for j in jsc
        copyto!(h[j], view(h∇, :, :, j))
    end
    # CUDA.synchronize()
    return nothing
end

seed!(123)
nbins = 32
nfeats = 100
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
h = [h∇[:,:,j] for j in axes(h∇, 3)]
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

CUDA.allowscalar(false)
@time update_hist_gpu2!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
CUDA.@time update_hist_gpu2!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
@time CUDA.@sync update_hist_gpu2!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
@btime update_hist_gpu2!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)




seed!(123)
nbins = 32
nfeats = 100
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
h = [h∇[:,:,j] for j in axes(h∇, 3)]
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

EvoTrees.update_hist_gpu!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)



seed!(123)
nbins = 32
nfeats = 1
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
h = [h∇[:,:,j] for j in axes(h∇, 3)]
rowsample = 0.5
colsample = 1.0
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

EvoTrees.update_hist_gpu!(h, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, js)
