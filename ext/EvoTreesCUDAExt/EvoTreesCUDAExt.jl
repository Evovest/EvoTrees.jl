module EvoTreesCUDAExt

using Base.Threads: @threads
using EvoTrees
using EvoTrees: split_set!, pred_leaf_cpu!, update_gains!, get_gain, Quantile
import EvoTrees: update_hist!
using CUDA

# This should be different on CPUs and GPUs
EvoTrees.device_ones(::Type{<:EvoTrees.GPU}, ::Type{T}, n::Int) where {T} = CUDA.ones(T, n)
EvoTrees.device_array_type(::Type{<:EvoTrees.GPU}) = CuArray
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.GPU})
    GC.gc(true)
    CUDA.reclaim()
end

include("loss.jl")
include("metrics.jl")
include("predict.jl")
include("structs.jl")
include("init.jl")
include("subsample.jl")
include("fit-utils.jl")
include("fit.jl")

end # module
