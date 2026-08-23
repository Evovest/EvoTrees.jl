# reproducible subsampling
function EvoTrees.subsample(is_full::CuVector, mask_cpu::Vector, mask_gpu::CuVector, rowsample::AbstractFloat, rng)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, mask_cpu)
    copyto!(mask_gpu, mask_cpu)
    is = is_full[mask_gpu.<=cond]
    return is
end

# Group-aware subsampling for ranking. One draw per group rather than per row, then a gather
# through the per-row group id to expand the group mask back to rows. This keeps the same
# boolean-mask indexing the row sampler uses, with a single gather on top, so a selected
# group always contributes all of its rows.
function EvoTrees.subsample(is_full::CuVector, mask_cpu::Vector, mask_gpu::CuVector, rowsample::AbstractFloat, rng, g::GroupCacheGPU)
    cond = round(UInt8, 255 * rowsample)
    rand!(rng, g.mask_cpu)
    copyto!(g.mask_gpu, g.mask_cpu)
    is = is_full[g.mask_gpu[g.group_gpu].<=cond]
    length(is) == 0 && error("no subsample group - choose larger rowsample")
    return is
end
