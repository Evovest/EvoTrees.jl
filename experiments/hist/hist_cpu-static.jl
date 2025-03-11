using Revise
# using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!
using StaticArrays

function hist_cpu_static!(
    hist::Matrix,
    ∇::Vector,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
)
    hist .*= 0
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[bin, j] += ∇[i]
        end
    end
    return nothing
end

nbins = 32
nobs = Int(1e6)
nfeats = 100
rowsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(SVector{3,Float32}, nobs);
h∇ = zeros(SVector{3,Float64}, nbins, nfeats)

####################################################
# vector
####################################################
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)

# laptop: 949.700 μs (41 allocations: 5.11 KiB)
# desktop: 468.078 μs (61 allocations: 6.52 KiB)
colsample = 0.01
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_static!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_static!($h∇, $∇, $x_bin, $is, $js)

# 2.788 ms (41 allocations: 5.11 KiB)
# desktop: 536.021 μs (61 allocations: 6.52 KiB)
colsample = 0.1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_static!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_static!($h∇, $∇, $x_bin, $is, $js)

# laptop: 23.854 ms (41 allocations: 5.11 KiB)
# desktop: 4.893 ms (61 allocations: 6.52 KiB)
colsample = 1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_static!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_static!($h∇, $∇, $x_bin, $is, $js)
