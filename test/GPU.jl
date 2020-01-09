# using CUDA
using CuArrays
using CUDAnative

features = rand(1_000, 10)
features_int = rand(UInt8, 1_000, 10)

nbins = 32
hist = zeros(Float32, nbins)
δ = rand(Float32, 1_000_000)
δ² = rand(Float32, 1_000_000)
idx = (rand(1:nbins, 1_000_000))
set = collect(1:length(idx))

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
idx_gpu = CuArray(idx)
set_gpu = CuArray(set)

function split_cpu(hist, δ, idx, set)
    @inbounds for i in set
        hist[idx[i]] += δ[i]
    end
    return
end

function split_gpu(hist, δ, idx, set)
    @inbounds for i in set
        hist[idx[i]] += δ[i]
    end
    return
end

hist
@time split_cpu(hist, δ, idx, set)
@time split_gpu(hist, δ_gpu, idx, set)

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

δ = rand(Float32, 1_000_000)
δ² = rand(Float32, 1_000_000)
δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
@time @cuda threads=256 gpu_add2!(δ_gpu, δ²_gpu)


function hist_test!(hist, idx, x)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(x)
        hist[idx[i]] += x[i]
    end
    return nothing
end

hist = zeros(Float32, nbins)
@time @cuda threads=256 hist_test!(hist_gpu, idx, δ_gpu)
