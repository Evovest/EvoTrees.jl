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

function update_hist_gpu!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
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
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    return nothing
end

seed!(123)
nbins = 32
nfeats = 100
nobs = Int(1e6)

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

x_bin_gpu = CuArray(x_bin)
∇_gpu = CuArray(∇)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
CUDA.@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)


"""
    hist_kernel!
    revserse k and tix thread order
"""
function hist_kernel!(h∇, ∇, x_bin, is, js)
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
                CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
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
    max_threads = config.threads ÷ 4
    max_blocks = config.blocks * 4
    tz = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, tz)))
    tx = max(1, min(length(is), fld(max_threads, tz * ty)))
    threads = (tz, ty, tx)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    return nothing
end

nbins = 32
nfeats = 100
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

x_bin_gpu = CuArray(x_bin)
∇_gpu = CuArray(∇)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
CUDA.@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)



"""
    revserse tix and tij thread order
    2.4 ms, no significant speed difference with previous k, j i order
"""
function hist_kernel!(h∇, ∇, x_bin, is, js)
    k, tix, tiy = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy = blockDim().y, blockDim().z
    bix, biy = blockIdx().y, blockIdx().z
    gdx = gridDim().y

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

function update_hist_gpu!(h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
    # @info "config.blocks" config.blocks
    max_threads = config.threads ÷ 4
    max_blocks = config.blocks * 4
    tk = size(h∇, 1)
    tj = max(1, min(length(js), fld(max_threads, tk)))
    ti = max(1, min(length(is), fld(max_threads, tk * tj)))
    threads = (tk, ti, tj)
    bj = cld(length(js), tj)
    bi = min(cld(max_blocks, bj), cld(length(is), ti))
    blocks = (1, bi, bj)
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    return nothing
end

nbins = 32
nfeats = 100
nobs = Int(1e6)
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float32, 3, nbins, nfeats)
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

x_bin_gpu = CuArray(x_bin)
∇_gpu = CuArray(∇)
h∇_gpu = CuArray(h∇)
is_gpu = CuArray(is)
js_gpu = CuArray(js)

@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
CUDA.@time update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@time CUDA.@sync update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
@btime update_hist_gpu!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu)
