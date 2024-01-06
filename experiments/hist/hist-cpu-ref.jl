using Revise
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads

function hist_cpu!(
    hist::Vector,
    ∇::Matrix,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
)
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[j][1, bin] += ∇[1, i]
            hist[j][2, bin] += ∇[2, i]
            hist[j][3, bin] += ∇[3, i]
        end
    end
    return nothing
end

nbins = 64
nobs = Int(1e6)
nfeats = 100
rowsample = 0.5
colsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = [zeros(Float64, 3, nbins) for n in 1:nfeats]
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
js = sample(1:nfeats, Int(round(rowsample * nfeats)), replace=false, ordered=true)

# laptop: 7.455 ms (81 allocations: 10.17 KiB)
# desktop: 3.381 ms (61 allocations: 6.52 KiB)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)
