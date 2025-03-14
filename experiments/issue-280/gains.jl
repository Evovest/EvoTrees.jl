using Revise
using CUDA
# using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!
using EvoTrees
using EvoTrees: EvoTypes, LossType, MSE, get_gain

function update_gains!(
    ::Type{L},
    _h,
    _hL,
    _hR,
    _gains,
    js,
    params::EvoTypes,
) where {L<:LossType}

    h = view(_h, :, :, js)
    hL = view(_hL, :, :, js)
    hR = view(_hR, :, :, js)
    gains = view(_gains, :, js)

    cumsum!(hL, h, dims=2)
    hR .= view(hL, :, size(hL, 2), 1) .- hL
    @inbounds for j in axes(h, 3)
        @inbounds for bin in axes(h, 2)
            gains[bin, j] =
                get_gain(L, params, view(hL, :, bin, j)) +
                get_gain(L, params, view(hR, :, bin, j))
        end
    end

    return nothing
end

nbins = 64
nfeats = 100
colsample = 0.5
js = sample(1:nfeats, Int(round(colsample * nfeats)), replace=false, ordered=true);

config = EvoTreeRegressor(; loss=:mse, nbins)
L = MSE

h = rand(3, nbins, nfeats)
hL = zeros(3, nbins, nfeats)
hR = zeros(3, nbins, nfeats)
gains = zeros(nbins, nfeats)

@time update_gains!(L, h, hL, hR, gains, js, config)
# 40.982 μs (0 allocations: 0 bytes)

@btime update_gains!(L, h, hL, hR, gains, js, config);
