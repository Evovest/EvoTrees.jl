module EvoTreesCUDAExt

using EvoTrees
using CUDA
using KernelAbstractions

EvoTrees.gpu_backend(::Type{<:EvoTrees.CUDADevice}) = CUDA.CUDABackend()
EvoTrees.device_array_type(::Type{<:EvoTrees.CUDADevice}) = CuArray
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.CUDADevice})
    GC.gc(true)
    CUDA.reclaim()
end

end

