using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using Random: seed!

"""
    hist_kernel!

Perform a single GPU hist per depth.
Use a vector of node indices. 
"""
function hist_kernel_single!(h∇::CuDeviceArray{T,4}, ∇::CuDeviceMatrix{S}, x_bin, is, js, ns) where {T,S}
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
                @inbounds ndx = ns[i]
                @inbounds idx = is[i]
                @inbounds bin = x_bin[idx, jdx]
                hid = Base._to_linear_index(h∇, k, bin, jdx, ndx)
                CUDA.atomic_add!(pointer(h∇, hid), T(∇[k, idx]))
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu_single!(h, h∇, ∇, x_bin, is, js, ns)
    kernel = @cuda launch = false hist_kernel_single!(h∇, ∇, x_bin, is, js, ns)
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
    kernel(h∇, ∇, x_bin, is, js, ns; threads, blocks)
    return nothing
end

seed!(123)
nbins = 32
nfeats = 100
nobs = Int(1e6)
max_depth = 6
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = [zeros(Float32, 3, nbins) for n in 1:nfeats]
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)
# ns = ones(Int, nobs)
nidx = rand(15:31, nobs)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
h∇_gpu = CUDA.zeros(Float32, 3, nbins, nfeats, 2^(max_depth - 1) - 1);
is_gpu = CuArray(is)
js_gpu = CuArray(js)
nidx_gpu = CuArray(nidx)

CUDA.allowscalar(false)
CUDA.@time update_hist_gpu_single!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, nidx_gpu)
# ref without copy to cpu: ~same
# ref 10K: 875.100 μs (168 allocations: 7.08 KiB)
# ref 100K: 1.236 ms (215 allocations: 9.91 KiB)
# ref 1M:  6.138 ms (227 allocations: 12.00 KiB)
# ref 10M: 67.075 ms (235 allocations: 13.38 KiB)
@btime update_hist_gpu_single!(h∇, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, ns_gpu)

function update_nodes_idx_gpu!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    for i in eachindex(is)
        n = nidx[i]
        feat = cond_feats[n]
        feattype = feattypes[feat]
        bin = cond_bins[n]
        is_right = feattype ? x_bin[is[i], feat] > bin : x_bin[is[i], feat] != bin
        nidx[i] = nidx[i] << 1 + is_right
    end
    return nothing
end

cond_feats = rand(js, nfeats)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)
update_nodes_idx_gpu!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
