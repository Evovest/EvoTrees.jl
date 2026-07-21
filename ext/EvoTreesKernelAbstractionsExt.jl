module EvoTreesKernelAbstractionsExt

using EvoTrees
using EvoTrees.Random: rand!, Xoshiro
using EvoTrees.Tables
using Atomix
using GPUArraysCore
using KernelAbstractions
using KernelAbstractions: get_backend

const KA = KernelAbstractions
const CuArray = GPUArraysCore.AbstractGPUArray
const CuVector{T} = GPUArraysCore.AbstractGPUArray{T,1}
const CuMatrix{T} = GPUArraysCore.AbstractGPUArray{T,2}
const AnyCuVector = GPUArraysCore.AbstractGPUArray{T,1} where {T}

_gpu_backend(device::Type{<:EvoTrees.GPU}) = EvoTrees.gpu_backend(device)
_to_device(backend, x::AbstractArray) =
    copyto!(KA.allocate(backend, eltype(x), size(x)), x)

EvoTrees.device_ones(device::Type{<:EvoTrees.GPU}, ::Type{T}, n::Int) where {T} =
    KA.ones(_gpu_backend(device), T, n)
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.GPU})
    GC.gc(true)
end

include("EvoTreesKernelAbstractionsExt/structs.jl")
include("EvoTreesKernelAbstractionsExt/loss.jl")
include("EvoTreesKernelAbstractionsExt/metrics.jl")
include("EvoTreesKernelAbstractionsExt/predict.jl")
include("EvoTreesKernelAbstractionsExt/init.jl")
include("EvoTreesKernelAbstractionsExt/subsample.jl")
include("EvoTreesKernelAbstractionsExt/fit-utils.jl")
include("EvoTreesKernelAbstractionsExt/fit.jl")

end
