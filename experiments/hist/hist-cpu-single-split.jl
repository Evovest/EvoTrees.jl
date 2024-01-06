using Revise
# using CUDA
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

h∇ = [[zeros(3, nbins) for nid in 1:(2^(max_depth-1)-1)] for nfeat in 1:nfeats];
# h∇ = [[zeros(3, nbins) for nfeat in 1:nfeats] for nid in 1:(2^(max_depth-1)-1)];
# gains = [[zeros(nbins) for nfeat in 1:nfeats] for nid in 1:(2^(max_depth-1)-1)];

cond_feats = rand(js, 2^(max_depth - 1) - 1)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)

function hist_single_cpu!(
    h∇::Vector{T},
    ∇::Matrix{S},
    x_bin::Matrix{UInt8},
    is::AbstractVector,
    js::AbstractVector,
    ns::AbstractVector,
) where {T,S}
    @threads for j in js
        _h∇ = h∇[j]
        @inbounds for i in is
            nid = ns[i]
            bin = x_bin[i, j]
            __h∇ = _h∇[nid]
            __h∇[1, bin] += ∇[1, i]
            __h∇[2, bin] += ∇[2, i]
            __h∇[3, bin] += ∇[3, i]
        end
    end
    return nothing
end

# laptop: 10.383 ms (81 allocations: 10.17 KiB)
# desktop:
@time hist_single_cpu!(h∇, ∇, x_bin, is, js, ns)
@btime hist_single_cpu!($h∇, $∇, $x_bin, $is, $js, $ns)

#####################################
# update gains
#####################################
const ϵ::Float64 = eps(eltype(Float64))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2

function update_gains_cpu_2!(gains, h∇L, h∇R, h∇, dnodes, lambda)
    for nid in dnodes
        for j in js
            # _gains, _h∇L, _h∇R, _h∇ = view(gains, :, j, nid), view(h∇L, :, :, j, nid), view(h∇R, :, :, j, nid), view(h∇, :, :, j, nid)
            # _gains = view(gains, :, j)
            _gains = gains[nid][j]
            # @views _gains = gains[:, j, nid]
            # cumsum!(_h∇L, _h∇; dims=2)
            # _h∇R .= _h∇L
            # reverse!(_h∇R; dims=2)
            # _h∇R .= view(_h∇R, :, 1:1) .- _h∇L
            # _gains .= get_gain.(view(_h∇L, 1, :), view(_h∇L, 2, :), view(_h∇L, 3, :), lambda) .+
            #           get_gain.(view(_h∇R, 1, :), view(_h∇R, 2, :), view(_h∇R, 3, :), lambda)

            # _gains .*= view(_h∇, 3, :) .!= 0
        end
    end
    return nothing
end

# gains = zeros(nbins, nfeats, 2^(max_depth - 1) - 1);
# gains = zeros(nbins, nfeats);
gains = [[zeros(nbins) for nfeat in 1:nfeats] for nid in 1:(2^(max_depth-1)-1)];

h∇L = zero(h∇);
h∇R = zero(h∇);
lambda = 0.1
dnodes = 16:31

@time update_gains_cpu_2!(
    gains,
    h∇L,
    h∇R,
    h∇,
    dnodes,
    lambda
)

# laptop : 2.195 ms (26 allocations: 38.55 KiB)
# desktop: 
@btime update_gains_cpu_2!(
    $gains,
    $h∇L,
    $h∇R,
    $h∇,
    $dnodes,
    $lambda
)

# laptop: 372.900 μs (12 allocations: 992 bytes)
# desktop: 
@time best = findmax(view(gains, :, :, dnodes); dims=(1, 2));
@btime findmax(view(gains, :, :, dnodes); dims=(1, 2));


#####################################
# update node index
#####################################
function update_nodes_idx_cpu!(ns_src, ns, is, x_bin, cond_feats, cond_bins, feattypes)
    @threads for i in is
        n = ns_src[i]
        feat = cond_feats[n]
        bin = cond_bins[n]
        feattype = feattypes[feat]
        is_left = feattype ? x_bin[i, feat] <= bin : x_bin[i, feat] == bin
        ns[i] = n << 1 + !is_left
    end
    return nothing
end

@time update_nodes_idx_cpu!(ns_src, ns, is, x_bin, cond_feats, cond_bins, feattypes)
# laptop - 941.000 μs (81 allocations: 10.42 KiB)
@btime update_nodes_idx_cpu!($ns_src, $ns, $is, $x_bin, $cond_feats, $cond_bins, $feattypes)
