using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using Random: seed!

T = Float64
seed!(123)
nbins = 64
nfeats = 100
nobs = Int(1e6)
max_depth = 6
x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
rowsample = 0.5
colsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)
nidx_cpu = Vector{UInt32}(rand(1:16, nobs))

∇_gpu = CuArray(∇)
x_bin_gpu = CuArray(x_bin)
# dimensions: [2 * K + 1, nbins, feats, nb_nodes]
h∇_cpu = zeros(T, 3, nbins, nfeats, 2^(max_depth - 1) - 1);
h∇_gpu = CuArray(h∇_cpu);
is_gpu = CuArray(is)
js_gpu = CuArray(js)
nidx_src = CuArray(nidx_cpu)
nidx_gpu = CuArray(nidx_cpu)

cond_feats = rand(js, 2^(max_depth - 1) - 1)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)

cond_feats_gpu = CuArray(cond_feats)
cond_bins_gpu = CuArray(cond_bins)
feattypes_gpu = CuArray(feattypes)


"""
    hist_kernel!

Perform a single GPU hist per depth.
Use a vector of node indices. 
"""
function hist_kernel_single_v1!(h∇::CuDeviceArray{T,4}, ∇::CuDeviceMatrix{S}, x_bin, is, js, ns) where {T,S}
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

function update_hist_gpu_single!(h∇, ∇, x_bin, is, js, ns)
    kernel = @cuda launch = false hist_kernel_single_v1!(h∇, ∇, x_bin, is, js, ns)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    k = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = min(64, max(1, min(length(is), fld(max_threads, k * ty))))
    threads = (k, ty, tx)
    max_blocks = min(65535, max_blocks * fld(max_threads, prod(threads)))
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js, ns; threads, blocks)
    CUDA.synchronize()
    return nothing
end

CUDA.allowscalar(false)
CUDA.@time update_hist_gpu_single!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, nidx_gpu)
# laptop - 1M - cpu copy - v1: 5.245 ms (162 allocations: 10.47 KiB)
# laptop - 1M - cpu copy - v2: 5.668 ms (164 allocations: 10.78 KiB)
# laptop - 1M - no copy - v1: 4.477 ms (113 allocations: 7.14 KiB)
@btime update_hist_gpu_single!(h∇_gpu, ∇_gpu, x_bin_gpu, is_gpu, js_gpu, nidx_gpu)


#####################################
# update gains
#####################################
function update_gains_gpu!(gains, h∇L_gpu, h∇R_gpu, h∇_gpu, lambda)
    cumsum!(h∇L_gpu, h∇_gpu; dims=2)
    h∇R_gpu .= h∇L_gpu
    reverse!(h∇R_gpu; dims=2)
    h∇R_gpu .= view(h∇R_gpu, :, 1:1, :, :) .- h∇L_gpu
    gains .= get_gain.(view(h∇L_gpu, 1, :, :, :), view(h∇L_gpu, 2, :, :, :), view(h∇L_gpu, 3, :, :, :), lambda) .+
             get_gain.(view(h∇R_gpu, 1, :, :, :), view(h∇R_gpu, 2, :, :, :), view(h∇R_gpu, 3, :, :, :), lambda)

    gains .*= view(h∇_gpu, 3, :, :, :) .!= 0
    return nothing
end

const ϵ::Float32 = eps(eltype(Float32))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2

gains = CUDA.zeros(nbins, nfeats, 2^(max_depth - 1) - 1);
h∇L_gpu = CUDA.zero(h∇_gpu);
h∇R_gpu = CUDA.zero(h∇_gpu);
lambda = 0.1f0
dnodes = 16:31
CUDA.@time update_gains_gpu!(
    view(gains, :, :, dnodes),
    view(h∇L_gpu, :, :, :, dnodes),
    view(h∇R_gpu, :, :, :, dnodes),
    view(h∇_gpu, :, :, :, dnodes),
    lambda)

# laptop - 1M:  103.000 μs (388 allocations: 39.94 KiB)
# @btime update_gains_gpu!(gains, h∇L_gpu, h∇R_gpu, h∇_gpu, lambda)
@btime update_gains_gpu!(
    view(gains, :, :, dnodes),
    view(h∇L_gpu, :, :, :, dnodes),
    view(h∇R_gpu, :, :, :, dnodes),
    view(h∇_gpu, :, :, :, dnodes),
    lambda)

best = findmax(view(gains, :, :, dnodes); dims=(1, 2));
CUDA.@time findmax(view(gains, :, :, dnodes); dims=(1, 2));

@btime best = findmax(view(gains, :, :, dnodes); dims=(1, 2));
@btime best = findmax(view(gains, :, :, :); dims=(1, 2));

# non-threaded: 176.900 μs (0 allocations: 0 bytes)
# with @threads: 31.800 μs (95 allocations: 10.36 KiB)
gains_cpu = Array(gains);
best_cpu = findmax(view(gains_cpu, :, :, dnodes); dims=(1, 2));
@time findmax(view(gains_cpu, :, :, dnodes); dims=(1, 2));
@btime findmax(view(gains_cpu, :, :, dnodes); dims=(1, 2));

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
            is_left = feattype ? x_bin[idx, feat] <= bin : x_bin[idx, feat] == bin
            nidx[idx] = n << 1 + is_left
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

CUDA.@time update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)

# laptop - 1M: 1.056 ms (114 allocations: 6.81 KiB)
@btime update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)
