module EvoTreesAMDGPUExt

using EvoTrees
using AMDGPU
using KernelAbstractions

EvoTrees.gpu_backend(::Type{<:EvoTrees.ROCmDevice}) = AMDGPU.ROCBackend()
EvoTrees.device_array_type(::Type{<:EvoTrees.ROCmDevice}) = AMDGPU.ROCArray

end
