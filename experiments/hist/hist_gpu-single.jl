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
        @inbounds for iter in 1:niter
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

# change iteration for threads to loop of adjacent is
function hist_kernel_single_v2!(h∇::CuDeviceArray{T,4}, ∇::CuDeviceMatrix{S}, x_bin, is, js, ns) where {T,S}
    tix, tiy, k = threadIdx().z, threadIdx().y, threadIdx().x
    bdx, bdy = blockDim().z, blockDim().y
    bix, biy = blockIdx().z, blockIdx().y
    gdx = gridDim().z

    j = tiy + bdy * (biy - 1)
    if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        i_ini = niter * (tix - 1) + niter * bdx * (bix - 1)
        @inbounds for iter in 1:niter
            i = i_ini + iter
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

function update_hist_gpu_single!(h∇_cpu, h∇, ∇, x_bin, is, js, ns)
    kernel = @cuda launch = false hist_kernel_single_v2!(h∇, ∇, x_bin, is, js, ns)
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
    kernel(h∇, ∇, x_bin, is, js, ns; threads, blocks)
    CUDA.synchronize()
    copyto!(h∇_cpu, h∇)
    return nothing
end

seed!(123)
nbins = 64
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
nidx = rand(15:31, nobs)

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
# dimensions: [2 * K + 1, nbins, feats, nb_nodes]
h∇_cpu = zeros(Float32, 3, nbins, nfeats, 2^(max_depth - 1) - 1);
h∇_gpu = CUDA.zeros(Float32, 3, nbins, nfeats, 2^(max_depth - 1) - 1);
is_gpu = CuArray(is)
js_gpu = CuArray(js)
nidx_gpu = CuArray(nidx)

CUDA.allowscalar(false)
CUDA.@time update_hist_gpu_single!(h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, nidx_gpu)

# laptop - 1M: 5.218 ms (120 allocations: 8.23 KiB)
@btime update_hist_gpu_single!(h∇_cpu, h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, nidx_gpu)


#####################################
# update node index
#####################################
function update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    tix = threadIdx().x
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(is)
    niter = cld(i_max, bdx * gdx)
    i_ini = niter * (tix - 1) + niter * bdx * (bix - 1)

    @inbounds for iter in 1:niter
        i = i_ini + iter
        if i <= i_max
            idx = is[i]
            n = nidx[idx]
            feat = cond_feats[n]
            bin = cond_bins[n]
            feattype = feattypes[feat]
            is_right = feattype ? x_bin[idx, feat] > bin : x_bin[idx, feat] != bin
            nidx[idx] = n << 1 + is_right
        end
    end
    sync_threads()
    return nothing
end

function update_nodes_idx_gpu!(nidx_src, nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    kernel = @cuda launch = false update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    threads = min(max_threads, length(is))
    blocks = min(max_blocks, cld(length(is), threads))
    nidx .= nidx_src
    kernel(nidx, is, x_bin, cond_feats, cond_bins, feattypes; threads, blocks)
    CUDA.synchronize()
    return nothing
end

cond_feats = rand(js, 2^(max_depth - 1) - 1)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)

cond_feats_gpu = CuArray(cond_feats)
cond_bins_gpu = CuArray(cond_bins)
feattypes_gpu = CuArray(feattypes)

nidx = rand(15:31, nobs)
nidx_gpu = CuArray(nidx)
nidx_src = CuArray(nidx)

CUDA.@time update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)
@btime update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)
