using Revise
using CUDA
# using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!
using EvoTrees

function create(nnodes, nbins, nfeats)
    h = [[ones(3, nbins) for feat in 1:nfeats] for node in 1:nnodes]
    return h
end

function zeroise_1!(h)
    @threads for n in eachindex(h)
        _h = h[n]
        for i in eachindex(_h)
            _h[i] .= 0
        end
    end
    return nothing
end

@time h = create(2048, 64, 100);
# 106.777 ms (206849 allocations: 320.52 MiB)
@btime h = create(2048, 64, 100);
h[5][5]

@time zeroise_1!(h)
# 15.443 ms (61 allocations: 6.14 KiB)
@btime zeroise_1!(h)


function create_2(nnodes, nbins, nfeats)
    # h = [ones(3, nbins, nfeats) for node in 1:nnodes]
    h = ones(3, nbins, nfeats, nnodes)
    return h
end

function zeroise_2!(h)
    h .= 0
    # @threads for n in eachindex(h)
    #     h[n] .= 0
    # end
    return nothing
end

@time h = create_2(2048, 64, 100);
# 18.737 ms (4097 allocations: 300.14 MiB)
@btime h = create_2(2048, 64, 100);
h[5]

@time zeroise_2!(h)
# 15.443 ms (61 allocations: 6.14 KiB)
@btime zeroise_2!(h)


nobs = Int(1e6)
nfeats = 100
x_bin = rand(UInt8, nobs, nfeats);

is_in = UInt32(1):UInt32(nobs)
is_out = zeros(UInt32, nobs)
mask_cond = zeros(UInt8, nobs)
mask_bool = zeros(Bool, nobs)
out = zeros(UInt32, nobs)
left = zeros(UInt32, nobs)
right = zeros(UInt32, nobs)

####################################################
# vector
####################################################
rowsample = 0.5
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)

# desktop: 1.790 ms (70 allocations: 3.89 MiB)
@btime _left, _right = EvoTrees.split_set!(
    mask_bool,
    is,
    x_bin,
    16,
    128,
    true,
);

# desktop: 1.677 ms (1 allocation: 96 bytes)
@btime _left, _right = EvoTrees.split_set_single!(
    is,
    x_bin,
    16,
    128,
    true,
    left,
    right,
    is_out,
    0,
);

# desktop: 189.627 μs (127 allocations: 15.06 KiB)
@btime _left, _right = EvoTrees.split_set_threads!(
    is_out,
    left,
    right,
    is,
    x_bin,
    16,
    128,
    true,
    0,
);


rowsample = 0.01
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
# deskptop: 7.211 μs (67 allocations: 19.31 KiB)
@btime _left, _right = EvoTrees.split_set!(
    mask_bool,
    is,
    x_bin,
    16,
    128,
    true,
);

# desktop: 910.618 ns (1 allocation: 96 bytes)
@btime _left, _right = EvoTrees.split_set_single!(
    is,
    x_bin,
    16,
    128,
    true,
    left,
    right,
    out,
    0,
);

# desktop: 11.560 μs (127 allocations: 14.69 KiB)
@btime _left3, _right3 = EvoTrees.split_set_threads!(
    is_out,
    left,
    right,
    is,
    x_bin,
    16,
    128,
    true,
    0,
);
