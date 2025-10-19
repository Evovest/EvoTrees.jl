using CUDA
using EvoTrees
using EvoTrees: subsample
using Random
using BenchmarkTools

nobs = 1_000_000
rowsample = 0.5
seed = 123

rng = Random.Xoshiro(seed)
x1 = zeros(UInt8, nobs);
@btime rand!(rng, x1);

x1 = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
@btime rand!(rng, x1);

x1 = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
@btime rand!(rng, x1);

# CPU
is = zeros(UInt32, nobs)
left = zeros(UInt32, nobs)
mask_cond = zeros(UInt8, nobs)
is_new = EvoTrees.subsample(left, is, mask_cond, rowsample, rng)
@btime EvoTrees.subsample(left, is, mask_cond, rowsample, rng);


function sub_cpu(is_in, mask, rowsample, rng)
    cond = round(UInt8, 255 * rowsample)
    Random.rand!(rng, mask)
    is = is_in[mask .<= cond]
    return is
end
is_in = collect(1:nobs)
mask = zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = sub_cpu(is_in, mask, rowsample, rng);
@btime sub_cpu(is_in, mask, rowsample, rng);

# GPU
is_in = CUDA.zeros(UInt32, nobs)
is_out = CUDA.zeros(UInt32, nobs)
mask = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = EvoTrees.subsample(is_in, is_out, mask, rowsample, rng);
@btime EvoTrees.subsample(is_in, is_out, mask, rowsample, rng);

function sub_gpu(is_in, mask, rowsample, rng)
    cond = round(UInt8, 255 * rowsample)
    CUDA.rand!(rng, mask)
    is = is_in[mask.<=cond]
    return is
end
is_in = CuArray{UInt32}(1:nobs)
rng = Random.Xoshiro(seed)
is_new = sub_gpu(is_in, mask, rowsample, rng);
@btime sub_gpu(is_in, mask, rowsample, rng);
