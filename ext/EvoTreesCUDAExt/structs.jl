"""
	CacheBaseGPU <: EvoTrees.CacheGPU

GPU training cache holding preallocated buffers used during tree growth.

### Quick reference (selected buffers)
- `h∇`: gradient histogram, indexed as `[2K+1, nbins, n_feats, node]`
- `nodes_sum_gpu`: per-node gradient totals `[2K+1, node]`
- `gains_per_feat_gpu`, `bins_per_feat_gpu`: per-(feature,node) best split results
- `best_gain_gpu`, `best_bin_gpu`, `best_feat_gpu`: reduced best split per node
- `split_sums_temp_gpu`: per-(node,feature) temporary buffer for K>1 split scanning
"""
struct CacheBaseGPU{Y,N<:EvoTrees.TrainNode} <: EvoTrees.CacheGPU
    rng::Xoshiro
    K::Int
    x_bin::CuMatrix{UInt8}
    y::Y
    w::Union{Nothing,CuVector}
    nodes::Vector{N}
    pred::CuMatrix
    nidx::CuVector{UInt32}
    is_full::CuVector{UInt32}
    mask_cpu::Vector{UInt8}
    mask_gpu::CuVector{UInt8}
    js_::Vector{UInt32}
    js::CuVector{UInt32}
    ∇::CuMatrix
    h∇::CuArray
    h∇L::CuArray
    h∇R::CuArray
    feature_names::Vector{Symbol}
    edges::Vector
    featbins::Vector
    feattypes_gpu::CuVector{Bool}
    cond_feats::Vector{UInt32}
    cond_feats_gpu::CuVector{UInt32}
    cond_bins::Vector{UInt8}
    cond_bins_gpu::CuVector{UInt8}
    monotone_constraints_gpu::CuVector{Int32}
    left_nodes_buf::CuVector{Int32}
    right_nodes_buf::CuVector{Int32}
    target_mask_buf::CuVector{UInt8}

    tree_split_gpu::CuVector{Bool}
    tree_cond_bin_gpu::CuVector{UInt8}
    tree_feat_gpu::CuVector{UInt32}
    tree_gain_gpu::CuVector{Float64}
    tree_pred_gpu::CuMatrix{Float32}
    nodes_sum_gpu::CuArray{Float64,2}
    anodes_gpu::CuVector{Int32}
    n_next_gpu::CuVector{Int32}
    n_next_active_gpu::CuVector{Int32}
    best_gain_gpu::CuVector{Float64}
    best_bin_gpu::CuVector{UInt8}
    best_feat_gpu::CuVector{UInt32}
    build_nodes_gpu::CuVector{Int32}
    subtract_nodes_gpu::CuVector{Int32}
    build_count::CuVector{Int32}
    subtract_count::CuVector{Int32}
    node_counts_gpu::CuVector{Int32}
    sums_temp_gpu::CuArray{Float64,2}        # Scratch: [2K+1, max_tree_nodes]
    gains_per_feat_gpu::CuMatrix{Float64}    # Output: best gain per (feature,node)  [n_sampled_feats, max_tree_nodes]
    bins_per_feat_gpu::CuMatrix{Int32}       # Output: best bin per (feature,node)   [n_sampled_feats, max_tree_nodes]
    split_sums_temp_gpu::CuMatrix{Float64}   # Temp: per-(node,feature) accumulators [2K+1, n_sampled_feats*max_tree_nodes]

end

