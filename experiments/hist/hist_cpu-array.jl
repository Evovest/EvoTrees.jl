using Revise
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!

function hist_cpu_2!(
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

# laptop: 949.700 μs (41 allocations: 5.11 KiB)
colsample = 0.01
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_2!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_2!($h∇, $∇, $x_bin, $is, $js)

# 2.788 ms (41 allocations: 5.11 KiB)
colsample = 0.1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_2!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_2!($h∇, $∇, $x_bin, $is, $js)

# laptop: 23.854 ms (41 allocations: 5.11 KiB)
colsample = 1
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true)
@time hist_cpu_2!(h∇, ∇, x_bin, is, js)
@btime hist_cpu_2!($h∇, $∇, $x_bin, $is, $js)
