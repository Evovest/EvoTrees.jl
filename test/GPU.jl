# using CUDA
using CUDAnative
using CuArrays
using GeometricFlux

features = rand(1_000, 10)
features_int = rand(UInt8, 1_000, 10)

nbins = 20
hist = zeros(Float32, 1, nbins)
δ = rand(Float32, 1, 10_000_000)
δ² = rand(Float32, 1, 10_000_000)
idx = rand(1:nbins, 10_000_000)
sub_idx = rand(1:nbins, 100_000)
set = collect(1:length(idx))
sub_set = sort(rand(set, 100_000))

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
idx_gpu = CuArray(idx)
set_gpu = CuArray(set)

@time sumpool(idx, δ)
@time sumpool(idx_gpu, δ_gpu)
@time scatter_add!(hist, δ, idx)
CuArrays.@time scatter_add!(hist_gpu, δ_gpu, idx_gpu)

CuArrays.@time CuArrays.scan!(+,δ²_gpu, δ_gpu,dims=1)

δ = rand(Float32, 1, 1_000)
hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ[1:1000])
idx_gpu = CuArray(idx[1:1000])
scatter_add!(hist_gpu, δ_gpu, idx_gpu)
sumpool(idx[1:1000], δ)

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
    index = threadIdx().x
    stride = blockDim().x
    @inbounds for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

δ = rand(Float32, 1_000_000)
δ² = rand(Float32, 1_000_000)
δ_gpu = CuArray(δ)
δ²_gpu = CuArray(δ²)
CuArrays.@time @cuda threads=256 gpu_add2!(δ_gpu, δ²_gpu)


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
