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
# 1M: 435.147 μs (126 allocations: 13.92 KiB)
# 10: 3.760 ms (126 allocations: 13.92 KiB)

function sub_cpu(is_in, mask, rowsample, rng)
    cond = round(UInt8, 255 * rowsample)
    Random.rand!(rng, mask)
    is = is_in[mask.<=cond]
    return is
end
is_in = collect(1:nobs)
mask = zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = sub_cpu(is_in, mask, rowsample, rng);
@btime sub_cpu(is_in, mask, rowsample, rng);
# 1M: 507.269 μs (6 allocations: 3.95 MiB)
# 10M: 19.315 ms (6 allocations: 39.61 MiB)

# GPU
is_in = CUDA.zeros(UInt32, nobs)
is_out = CUDA.zeros(UInt32, nobs)
mask = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = EvoTrees.subsample(is_in, is_out, mask, rowsample, rng);
@btime EvoTrees.subsample(is_in, is_out, mask, rowsample, rng);
# 1M: 280.630 μs (628 allocations: 19.36 KiB)
# 10M: 1.832 ms (630 allocations: 19.39 KiB)

# basic approach
function sub_gpu_v1(is_in, mask, rowsample, rng)
    cond = round(UInt8, 255 * rowsample)
    CUDA.rand!(rng, mask)
    is = is_in[mask.<=cond]
    return is
end
is_in = CuArray{UInt32}(1:nobs)
mask = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = sub_gpu_v1(is_in, mask, rowsample, rng);
@btime sub_gpu_v1(is_in, mask, rowsample, rng);
# 1M: 364.493 μs (928 allocations: 1005.06 KiB)
# 10M: 2.727 ms (931 allocations: 9.56 MiB)

# generate random on cpu
function sub_gpu_v2(is_full, mask_cpu, mask_gpu, rowsample, rng)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, mask_cpu)
    copyto!(mask_gpu, mask_cpu)
    # is = view(is_full, mask_gpu .<= cond)
    is = is_full[mask_gpu .<= cond]
    return is
end
is_full = CuArray{UInt32}(1:nobs)
mask_cpu = zeros(UInt8, nobs)
mask_gpu = CUDA.zeros(UInt8, nobs)
rng = Random.Xoshiro(seed)
is_new = sub_gpu_v2(is_full, mask_cpu, mask_gpu, rowsample, rng);
@btime sub_gpu_v2(is_full, mask_cpu, mask_gpu, rowsample, rng);
# 1M: 337.523 μs (926 allocations: 28.39 KiB)
# 10M: 2.363 ms (853 allocations: 26.38 KiB)
