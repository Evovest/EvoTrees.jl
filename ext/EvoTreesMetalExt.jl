module EvoTreesMetalExt

using EvoTrees
using KernelAbstractions
using Metal

EvoTrees.gpu_backend(::Type{<:EvoTrees.MetalDevice}) = Metal.MetalBackend()
EvoTrees.device_array_type(::Type{<:EvoTrees.MetalDevice}) = Metal.MtlArray

end
