using Revise
using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using Random: seed!

seed!(123)

T∇ = Float32
Th = Float64
nbins = 64
nfeats = 100
nobs = Int(1e6)
max_depth = 6
K = 3
rowsample = 0.5
colsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(SVector{3,T∇}, nobs);
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)
dnodes = 16:31
ns = Vector{UInt32}(rand(dnodes, nobs))
ns_src = copy(ns)
h∇ = zeros(SVector{3,Th}, nbins, nfeats, 2^(max_depth - 1) - 1);
h∇L = zero(h∇);
h∇R = zero(h∇);
gains = zeros(nbins, nfeats, 2^(max_depth - 1) - 1);
lambda = 0.1

cond_feats = rand(js, 2^(max_depth - 1) - 1)
cond_bins = rand(1:nbins, 2^(max_depth - 1) - 1)
feattypes = ones(Bool, nfeats)

function mutate!(d)
    for i in eachindex(d)
        d1, d2, d3 = rand(), rand(), rand()
        d1, d2, d3 = rand(), rand(), rand()
        d[i] = SVector{3, Float32}(d1, d2, d3)
    end
    return nothing
end
∇ = rand(SVector{3,T∇}, nobs);
@time mutate!(∇)
@btime mutate!(∇)

function hist_single_cpu!(
    h∇::Array{T,3},
    ∇::Vector{S},
    x_bin::Matrix{UInt8},
    is::AbstractVector,
    js::AbstractVector,
    ns::AbstractVector,
) where {T,S}
    @threads for j in js
        @inbounds for i in is
            n = ns[i]
            if n != 0
                bin = x_bin[i, j]
                h∇[bin, j, n] += ∇[i]
            end
        end
    end
    return nothing
end

# laptop: 10.383 ms (81 allocations: 10.17 KiB)
# desktop:
@time hist_single_cpu!(h∇, ∇, x_bin, is, js, ns)
@btime hist_single_cpu!($h∇, $∇, $x_bin, $is, $js, $ns)

#####################################
# update gains - ref
#####################################
const ϵ::Float64 = eps(eltype(Float64))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2

# function update_gains_cpu!(gains, h∇L, h∇R, h∇, lambda)
#     cumsum!(h∇L, h∇; dims=2)
#     h∇R .= h∇L
#     reverse!(h∇R; dims=2)
#     h∇R .= view(h∇R, :, 1:1, :, :) .- h∇L
#     gains .= get_gain.(view(h∇L, 1, :, :, :), view(h∇L, 2, :, :, :), view(h∇L, 3, :, :, :), lambda) .+
#              get_gain.(view(h∇R, 1, :, :, :), view(h∇R, 2, :, :, :), view(h∇R, 3, :, :, :), lambda)

#     gains .*= view(h∇, 3, :, :, :) .!= 0
#     return nothing
# end

# @time update_gains_cpu!(
#     view(gains, :, :, dnodes),
#     view(h∇L, :, :, :, dnodes),
#     view(h∇R, :, :, :, dnodes),
#     view(h∇, :, :, :, dnodes),
#     lambda)

# # laptop : 2.195 ms (26 allocations: 38.55 KiB)
# # desktop: 
# @btime update_gains_cpu!(
#     view(gains, :, :, dnodes),
#     view(h∇L, :, :, :, dnodes),
#     view(h∇R, :, :, :, dnodes),
#     view(h∇, :, :, :, dnodes),
#     lambda)

# # laptop: 372.900 μs (12 allocations: 992 bytes)
# # desktop: 
# @time best = findmax(view(gains, :, :, dnodes); dims=(1, 2));
# @btime findmax(view(gains, :, :, dnodes); dims=(1, 2));

#####################################
# update gains - threaded
#####################################
function update_gains_cpu_2!(gains, h∇L, h∇R, h∇, dnodes, lambda)
    @threads for j in js
        @inbounds for nid in dnodes
            _gains, _h∇L, _h∇R, _h∇ = view(gains, :, j, nid), view(h∇L, :, :, j, nid), view(h∇R, :, :, j, nid), view(h∇, :, :, j, nid)
            cumsum!(_h∇L, _h∇; dims=2)
            _h∇R .= _h∇L
            reverse!(_h∇R; dims=2)
            _h∇R .= view(_h∇R, :, 1:1) .- _h∇L
            _gains .= get_gain.(view(_h∇L, 1, :), view(_h∇L, 2, :), view(_h∇L, 3, :), lambda) .+
                      get_gain.(view(_h∇R, 1, :), view(_h∇R, 2, :), view(_h∇R, 3, :), lambda)

            _gains .*= view(_h∇, 3, :) .!= 0
        end
    end
    return nothing
end

h∇L = zero(h∇);
h∇R = zero(h∇);

@time update_gains_cpu_2!(
    gains,
    h∇L,
    h∇R,
    h∇,
    dnodes,
    lambda
)

# laptop : 102.200 μs (1685 allocations: 123.12 KiB)
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
        if n == 0
            ns[i] = 0
        else
            feat = cond_feats[n]
            cbin = cond_bins[n]
            if cbin == 0
                ns[i] = 0
            else
                feattype = feattypes[feat]
                is_left = feattype ? x_bin[i, feat] <= cbin : x_bin[i, feat] == cbin
                ns[i] = n << 1 + !is_left
            end
        end
    end
    return nothing
end

@time update_nodes_idx_cpu!(ns_src, ns, is, x_bin, cond_feats, cond_bins, feattypes)
# laptop - 1.028 ms (81 allocations: 10.42 KiB)
@btime update_nodes_idx_cpu!($ns_src, $ns, $is, $x_bin, $cond_feats, $cond_bins, $feattypes)
