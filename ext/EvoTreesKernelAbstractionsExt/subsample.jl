# reproducible subsampling
function EvoTrees.subsample(is_full::CuVector, mask_cpu::Vector, mask_gpu::CuVector, rowsample::AbstractFloat, rng)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, mask_cpu)
    copyto!(mask_gpu, mask_cpu)
    is = is_full[mask_gpu.<=cond]
    return is
end

# Group-aware subsampling: one draw per group, then a gather through the per-row group id to
# expand the mask back to rows. Same boolean-mask indexing as above, with one gather on top.
function EvoTrees.subsample(is_full::CuVector, mask_cpu::Vector, mask_gpu::CuVector, rowsample::AbstractFloat, rng, g::GroupCacheGPU)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, g.mask_cpu)
    while all(>(cond), g.mask_cpu)
        rand!(rng, g.mask_cpu)
    end
    copyto!(g.mask_gpu, g.mask_cpu)
    return is_full[g.mask_gpu[g.group_gpu].<=cond]
end
