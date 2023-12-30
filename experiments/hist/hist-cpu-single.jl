using Revise
using CUDA
# using StaticArrays
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
ns = Vector{UInt32}(rand(1:16, nobs))
ns_src = copy(ns)
h∇ = zeros(T, 3, nbins, nfeats, 2^(max_depth - 1) - 1);

cond_feats = rand(js, 2^(max_depth - 1) - 1)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)

function hist_single_cpu!(
    h∇::Array{T,4},
    ∇::Matrix{S},
    x_bin::Matrix{UInt8},
    is::AbstractVector,
    js::AbstractVector,
    ns::AbstractVector,
) where {T,S}
    @threads for j in js
        @inbounds for i in is
            bin = x_bin[i, j]
            ndx = ns[i]
            # @views h∇[:, bin, j, ndx] .+= ∇[:, i]
            h∇[1, bin, j, ndx] += ∇[1, i]
            h∇[2, bin, j, ndx] += ∇[2, i]
            h∇[3, bin, j, ndx] += ∇[3, i]
        end
    end
    return nothing
end

function grads!(x, y)
    @simd for k in eachindex(x)
        x[k] += y[k]
    end
    return nothing
end

# laptop: 6.886 ms (97 allocations: 10.67 KiB)
# desktop:
@time hist_single_cpu!(h∇, ∇, x_bin, is, js, ns)
@btime hist_single_cpu!($h∇, $∇, $x_bin, $is, $js, $ns)


#####################################
# update gains
#####################################
function update_gains_cpu!(gains, h∇L, h∇R, h∇, lambda)
    cumsum!(h∇L, h∇; dims=2)
    h∇R .= h∇L
    reverse!(h∇R; dims=2)
    h∇R .= view(h∇R, :, 1:1, :, :) .- h∇L
    gains .= get_gain.(view(h∇L, 1, :, :, :), view(h∇L, 2, :, :, :), view(h∇L, 3, :, :, :), lambda) .+
             get_gain.(view(h∇R, 1, :, :, :), view(h∇R, 2, :, :, :), view(h∇R, 3, :, :, :), lambda)

    gains .*= view(h∇, 3, :, :, :) .!= 0
    return nothing
end

const ϵ::Float64 = eps(eltype(Float64))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2

gains = zeros(nbins, nfeats, 2^(max_depth - 1) - 1);
h∇L = zero(h∇);
h∇R = zero(h∇);
lambda = 0.1
dnodes = 16:31
@time update_gains_cpu!(
    view(gains, :, :, dnodes),
    view(h∇L, :, :, :, dnodes),
    view(h∇R, :, :, :, dnodes),
    view(h∇, :, :, :, dnodes),
    lambda)

# laptop : 2.195 ms (26 allocations: 38.55 KiB)
# desktop: 
@btime update_gains_cpu!(
    view(gains, :, :, dnodes),
    view(h∇L, :, :, :, dnodes),
    view(h∇R, :, :, :, dnodes),
    view(h∇, :, :, :, dnodes),
    lambda)

# laptop: 372.900 μs (12 allocations: 992 bytes)
# desktop: 
@time best = findmax(view(gains, :, :, dnodes); dims=(1, 2));
@btime findmax(view(gains, :, :, dnodes); dims=(1, 2));

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
            nidx[idx] = n << 1 + !is_left
        end
    end
    sync_threads()
    return nothing
end

function update_nodes_idx_cpu!(ns_src, ns, is, x_bin, cond_feats, cond_bins, feattypes)
    @inbounds for i in is
        n = ns_src[idx]
        feat = cond_feats[n]
        bin = cond_bins[n]
        feattype = feattypes[feat]
        is_left = feattype ? x_bin[i, feat] <= bin : x_bin[i, feat] == bin
        ns[i] = n << 1 + !is_left
    end
    return nothing
end

CUDA.@time update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)

# laptop - 1M: 1.056 ms (114 allocations: 6.81 KiB)
@btime update_nodes_idx_gpu!(nidx_src, nidx_gpu, is_gpu, x_bin_gpu, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)
