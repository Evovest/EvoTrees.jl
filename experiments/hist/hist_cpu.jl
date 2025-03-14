using Revise
using CUDA
# using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!

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

nbins = 32
nobs = Int(1e6)
nfeats = 100
rowsample = 0.5

x_bin = UInt8.(rand(1:nbins, nobs, nfeats));
∇ = rand(Float32, 3, nobs);
h∇ = [zeros(Float64, 3, nbins) for n in 1:nfeats]

####################################################
# vector
####################################################
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)

# laptop: 571.700 μs (41 allocations: 5.11 KiB)
# desktop: 632.805 μs (61 allocations: 6.52 KiB)
colsample = 0.01
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# 1.397 ms (41 allocations: 5.11 KiB)
# desktop: 714.998 μs (61 allocations: 6.52 KiB)
colsample = 0.1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# laptop: 15.139 ms (41 allocations: 5.11 KiB)
# desktop: 6.411 ms (61 allocations: 6.52 KiB)
colsample = 1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

####################################################
# UnitRange - little slower
####################################################
mask = rand(Bool, nobs)
is = view(1:nobs, mask)

# laptop: 998.000 μs (41 allocations: 5.36 KiB)
colsample = 0.01
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# 3.005 ms (41 allocations: 5.36 KiB)
colsample = 0.1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# laptop: 28.263 ms (41 allocations: 5.36 KiB)
colsample = 1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

####################################################
# continuous view - as fast as vector (view(::Vector{Int64}, 1:499431))
####################################################
mask = rand(Bool, nobs)
_is = view(1:nobs, mask)
is = zeros(Int, nobs)
is[1:length(_is)] .= _is
is = view(is, 1:length(_is))

# laptop: 921.700 μs (41 allocations: 5.36 KiB)
colsample = 0.01
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# laptop: 2.660 ms (41 allocations: 5.36 KiB)
colsample = 0.1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)

# laptop: 23.646 ms (41 allocations: 5.36 KiB)
colsample = 1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu!(h∇, ∇, x_bin, is, js)
@btime hist_cpu!($h∇, $∇, $x_bin, $is, $js)
