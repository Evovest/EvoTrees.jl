using Revise
using CUDA
using StatsBase: sample
using BenchmarkTools
using Random: seed!

function convert_kernel!(x, y)
    i = threadIdx().x
    if i <= length(x)
        val = Float64(y[i])
        CUDA.atomic_add!(pointer(x, i), val)
        # CUDA.atomic_add!(pointer(x, i), y[i])
    end
    sync_threads()
    return nothing
end

function convert_gpu!(x, y)
    @cuda threads = length(x) blocks = (1,) convert_kernel!(x, y)
    CUDA.synchronize()
    return nothing
end

x = CUDA.zeros(Float32, 3)
y = CUDA.rand(Float32, 3)
convert_gpu!(x, y)
