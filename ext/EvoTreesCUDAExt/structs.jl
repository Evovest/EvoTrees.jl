struct CacheGPU
    info::Dict
    x_bin::CuMatrix
    y::CuArray
    w::Union{Nothing, CuVector}
    K::Int
    nodes::Union{Vector, Nothing}
    pred::CuMatrix
    nidx::CuVector{UInt32}
    is_in::CuVector{UInt32}
    is_out::CuVector{UInt32}
    mask::CuVector{UInt8}
    js_::Vector{UInt32}
    js::CuVector{UInt32}
    ∇::CuMatrix
    h∇::CuArray
    h∇L::Union{Nothing, CuArray}  
    h∇R::Union{Nothing, CuArray}  
    fnames::Vector{Symbol}  
    edges::Vector
    featbins::Vector
    feattypes_gpu::CuVector{Bool}
    cond_feats::Union{Nothing, Vector{Int}}  
    cond_feats_gpu::Union{Nothing, CuVector}  
    cond_bins::Union{Nothing, Vector{UInt8}}  
    cond_bins_gpu::Union{Nothing, CuVector}  
    monotone_constraints_gpu::CuVector{Int32}
    left_nodes_buf::CuVector{Int32}
    right_nodes_buf::CuVector{Int32}
    target_mask_buf::CuVector{UInt8}
    
    tree_split_gpu::CuVector{Bool}
    tree_cond_bin_gpu::CuVector{UInt8}
    tree_feat_gpu::CuVector{Int32}
    tree_gain_gpu::CuVector{Float64}
    tree_pred_gpu::CuMatrix{Float32}
    nodes_sum_gpu::CuArray{Float32,2}  
    nodes_gain_gpu::CuVector{Float32}
    anodes_gpu::CuVector{Int32}
    n_next_gpu::CuVector{Int32}
    n_next_active_gpu::CuVector{Int32}
    best_gain_gpu::CuVector{Float32}
    best_bin_gpu::CuVector{Int32}
    best_feat_gpu::CuVector{Int32}
    build_nodes_gpu::CuVector{Int32}
    subtract_nodes_gpu::CuVector{Int32}
    build_count::CuVector{Int32}
    subtract_count::CuVector{Int32}
    pre_leaf_gpu::CuVector{Float32}
end
