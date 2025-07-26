using Revise
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!
using FHist

function hist_cpu_array!(
    hist::Array{Float64,3},
    ∇::Matrix,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
)
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[1, i]
            hist[2, bin, j] += ∇[2, i]
            hist[3, bin, j] += ∇[3, i]
        end
    end
    return nothing
end

nbins = 32
nobs = Int(1e6)
nfeats = 100
rowsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = zeros(Float64, 3, nbins, nfeats)

####################################################
# vector
####################################################
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
# laptop: 18.576 ms (41 allocations: 5.11 KiB)
js = 1:nfeats
@time hist_cpu_array!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_array!($h∇, $∇, $x_bin, $is, $js)

####################################################
# FHist
####################################################
binedges = collect((0:(nbins)) ./ (nbins))
feats = rand(nobs)[is]
# laptop: 20.394 ms (14 allocations: 1.97 KiB)
@time bins = Hist1D(feats; binedges, overflow=true)
@btime Hist1D($feats; binedges=$binedges, overflow=true)

