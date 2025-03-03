using Revise
using CUDA
# using StaticArrays
using StatsBase: sample
using BenchmarkTools
using Base.Threads: @threads
using StatsBase: sample!
using EvoTrees

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
