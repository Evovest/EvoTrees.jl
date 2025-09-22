module EvoTreesCUDAExt

using EvoTrees
using CUDA
using KernelAbstractions
using Atomix
using Adapt
using Tables
using KernelAbstractions: get_backend

EvoTrees.device_ones(::Type{<:EvoTrees.GPU}, ::Type{T}, n::Int) where {T} = CUDA.ones(T, n)
EvoTrees.device_array_type(::Type{<:EvoTrees.GPU}) = CuArray
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.GPU})
    GC.gc(true)
    CUDA.reclaim()
end

include("structs.jl")
include("loss.jl")
include("metrics.jl")
include("predict.jl")
include("init.jl")
include("subsample.jl")
include("fit-utils.jl")
include("fit.jl")

end

