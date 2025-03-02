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

# laptop: 2.881 ms (50 allocations: 3.88 MiB)
@btime _left, _right = EvoTrees.split_set!(
    mask_bool,
    is,
    x_bin,
    16,
    128,
    true,
);

# laptop: 76.136 ns (1 allocation: 96 bytes)
@btime _left, _right = EvoTrees.split_set_single!(
    is,
    x_bin,
    16,
    128,
    true,
    left,
    right
);

# laptop: 2.009 ms (5 allocations: 608 bytes)
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


rowsample = 0.001
is = sample(1:nobs, Int(round(rowsample * nobs)), replace=false, ordered=true)
# laptop: 8.400 μs (51 allocations: 18.03 KiB)
@btime _left, _right = EvoTrees.split_set!(
    mask_bool,
    is,
    x_bin,
    16,
    128,
    true,
);

# laptop: 76.136 ns (1 allocation: 96 bytes)
@btime _left, _right = EvoTrees.split_set_single!(
    is,
    x_bin,
    16,
    128,
    true,
    left,
    right
);

# laoptop: 189.533 ns (5 allocations: 352 bytes)
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
