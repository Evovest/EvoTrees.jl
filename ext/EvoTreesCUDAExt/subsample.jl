# reproducible subsampling
function EvoTrees.subsample(is_full::CuVector, mask_cpu::Vector, mask_gpu::CuVector, rowsample::AbstractFloat, rng)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, mask_cpu)
    copyto!(mask_gpu, mask_cpu)
    is = is_full[mask_gpu.<=cond]
    # is = view(is_full, mask_gpu.<=cond)
    return is
end
