function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        # Feature sampling done on CPU then copied to GPU
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow_tree!(tree, params, cache, is)
        push!(evotree.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end
    
    evotree.info[:nrounds] += 1
    return nothing
end

function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}
    @warn "Oblivious tree GPU implementation not yet available, using standard tree" maxlog=1
    grow_tree!(tree, params, cache, is)
end

# Grow decision tree level-by-level with histogram subtraction optimization
# Tree structure: root=1, left_child=2*parent, right_child=2*parent+1
# Key optimization: at depth≥2, build histogram for smaller child, subtract for larger
function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}

    backend = KernelAbstractions.get_backend(cache.x_bin)

    ∇_gpu = cache.∇
    if L <: EvoTrees.MAE
        ∇_gpu = copy(cache.∇)
        ∇_gpu[2, :] .= 1.0f0
    end

    # Initialize cache arrays
    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0
    cache.tree_pred_gpu .= 0
    cache.nodes_sum_gpu .= 0
    cache.anodes_gpu .= 0
    cache.n_next_gpu .= 0
    cache.n_next_active_gpu .= 0
    cache.best_gain_gpu .= 0
    cache.best_bin_gpu .= 0
    cache.best_feat_gpu .= 0
    cache.nidx .= 1
    view(cache.anodes_gpu, 1:1) .= 1

    # Root node processing
    if params.max_depth == 1
        reduce_root_sums_kernel!(backend)(
            cache.nodes_sum_gpu, ∇_gpu, is; 
            ndrange=length(is), 
            workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)
    else
        update_hist_gpu!(
            L, cache.h∇, ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
            1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
            cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, 
            Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:1),
            cache.target_mask_buf, backend
        )
        
        find_split_root! = find_best_split_from_hist_kernel!(backend)
        find_split_root!(
            L, view(cache.best_gain_gpu, 1:1), 
            view(cache.best_bin_gpu, 1:1), 
            view(cache.best_feat_gpu, 1:1),
            cache.h∇, cache.nodes_sum_gpu,
            view(cache.anodes_gpu, 1:1),
            cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
            Float32(params.lambda), Float32(params.L2), Float32(params.min_weight),
            cache.K, view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:1);
            ndrange = 1, 
            workgroupsize = 64
        )
        KernelAbstractions.synchronize(backend)
    end

    n_active = params.max_depth == 1 ? 0 : 1

    # Main loop: build tree level by level
    for depth in 1:params.max_depth
        iszero(n_active) && break
        
        view(cache.n_next_active_gpu, 1:1) .= 0
        n_nodes = 2^(depth - 1)
        active_nodes = view(cache.anodes_gpu, 1:n_nodes)
        
        if n_active < n_nodes
            view(cache.anodes_gpu, n_active+1:n_nodes) .= 0
        end

        # ====================================================================
        # HISTOGRAM SUBTRACTION OPTIMIZATION (depth ≥ 2)
        # ====================================================================
        # At depth ≥ 2, parent histograms are available, so we use subtraction
        # Key insight: h∇[parent] = h∇[left_child] + h∇[right_child]
        # Therefore: h∇[larger_child] = h∇[parent] - h∇[smaller_child]
        if depth >= 2
            # Clear tracking arrays
            cache.build_nodes_gpu .= 0
            cache.subtract_nodes_gpu .= 0
            cache.build_count .= 0
            cache.subtract_count .= 0
            
            # STEP 1: Separate active_nodes into BUILD and SUBTRACT lists
            # active_nodes contains all children at current depth
            # nodes_sum_gpu contains parent gradient sums (needed for weight comparison)
            separate_kernel! = separate_nodes_kernel!(backend)
            separate_kernel!(
                cache.build_nodes_gpu, cache.build_count,        # Output: smaller children
                cache.subtract_nodes_gpu, cache.subtract_count,  # Output: larger children
                view(active_nodes, 1:n_active),                  # Input: all active children
                cache.nodes_sum_gpu,                             # Input: has parent sums
                cache.K;                                         # For weight index calc
                ndrange = n_active,
                workgroupsize = 256
            )
            KernelAbstractions.synchronize(backend)
            
            # STEP 2: Get counts from GPU
            build_count_val = Array(cache.build_count)[1]        # How many to BUILD
            subtract_count_val = Array(cache.subtract_count)[1]  # How many to SUBTRACT
            
            # STEP 3: Build histograms only for smaller children
            # This is the expensive part - scanning observations
            if build_count_val > 0
                update_hist_gpu!(
                    L, cache.h∇, ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), 
                    cache.nodes_sum_gpu, params,
                    cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, 
                    Float32(params.L2), 
                    view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:max(build_count_val, 1)),
                    cache.target_mask_buf, backend
                )
            end
            
            # STEP 4: Compute larger children via subtraction
            # This is FAST - pure arithmetic, no observation scan!
            # For each node in subtract_nodes: h∇[node] = h∇[parent] - h∇[sibling]
            if subtract_count_val > 0
                subtract_hist_kernel!(backend)(
                    cache.h∇,                                             # Histogram to update
                    view(cache.subtract_nodes_gpu, 1:subtract_count_val); # Nodes to compute
                    ndrange = subtract_count_val * size(cache.h∇, 1) * size(cache.h∇, 2) * size(cache.h∇, 3),
                    workgroupsize = 256
                )
                KernelAbstractions.synchronize(backend)
            end
            
            # STEP 5: Find best splits for ALL active nodes
            # Now all histograms are ready (some built, some subtracted)
            # We pass the original active_nodes to maintain correct indexing
            find_split_all! = find_best_split_from_hist_kernel!(backend)
            find_split_all!(
                L, view(cache.best_gain_gpu, 1:n_nodes), 
                view(cache.best_bin_gpu, 1:n_nodes), 
                view(cache.best_feat_gpu, 1:n_nodes),
                cache.h∇, cache.nodes_sum_gpu,
                view(active_nodes, 1:n_active), 
                cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
                Float32(params.lambda), Float32(params.L2), Float32(params.min_weight),
                cache.K, view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:n_active);
                ndrange = n_active, 
                workgroupsize = min(256, max(64, n_active))
            )
            KernelAbstractions.synchronize(backend)
        end

        # Apply splits: create children if gain > threshold, else make leaf
        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, 
            cache.tree_gain_gpu, cache.tree_pred_gpu, cache.nodes_sum_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view(cache.best_gain_gpu, 1:n_nodes), 
            view(cache.best_bin_gpu, 1:n_nodes), 
            view(cache.best_feat_gpu, 1:n_nodes),
            cache.h∇, active_nodes, cache.feattypes_gpu,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), 
            Float32(params.L2), cache.K;
            ndrange = max(n_active, 1), 
            workgroupsize = 256
        )
        KernelAbstractions.synchronize(backend)
        
        n_active = Array(cache.n_next_active_gpu)[1]
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        # Update observation→node assignments for next level
        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, 
                cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), 
                workgroupsize = 256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Copy tree to CPU and compute leaf predictions
    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))

    leaf_nodes = findall(!, tree.split)

    if L <: Union{EvoTrees.MAE, EvoTrees.Quantile}
        # Special handling: MAE/Quantile need median computation on CPU
        cpu_data = (
            nidx = Array(cache.nidx),
            is = Array(is),
            ∇ = Array(cache.∇),
            nodes_sum = Array(cache.nodes_sum_gpu)
        )
        
        leaf_map = Dict{Int, Vector{UInt32}}()
        sizehint!(leaf_map, length(leaf_nodes))
        for i in 1:length(cpu_data.is)
            leaf_id = cpu_data.nidx[cpu_data.is[i]]
            if !haskey(leaf_map, leaf_id)
                leaf_map[leaf_id] = UInt32[]
            end
            push!(leaf_map[leaf_id], cpu_data.is[i])
        end
        
        for n in leaf_nodes
            node_sum_view = view(cpu_data.nodes_sum, :, n)
            if L <: EvoTrees.Quantile
                node_is = get(leaf_map, n, UInt32[])
                if !isempty(node_is)
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params, cpu_data.∇, node_is)
                else
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, EvoTrees.MAE, params)
                end
            else
                EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params)
            end
        end
    else
        # Standard loss: pred = -g / (h + λ*w + L2)
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        for n in leaf_nodes
            node_sum_view = view(nodes_sum_cpu, :, n)
            EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params)
        end
    end
    
    return nothing
end

# Apply splits kernel: decide split vs leaf, compute child gradients/predictions
# Child gradient sums extracted from histogram enable subtraction at next level
@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, n_next, n_next_active,
    best_gain, best_bin, best_feat, h∇, active_nodes, feattypes,
    depth, max_depth, lambda, gamma, L2, K_val
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]
    eps = eltype(tree_pred)(1e-8)

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l = node << 1
        child_r = (node << 1) + 1
        feat = Int(tree_feat[node])
        bin = Int(tree_cond_bin[node])
        is_numeric = feattypes[feat]

        # Compute child gradient sums from histogram (enables subtraction at next level!)
        # This is CRITICAL: stores gradient sums so next level can use:
        #   h∇[grandchild] = h∇[parent=this_node] - h∇[sibling]
        for kk in 1:(2*K_val+1)
            sum_val = zero(eltype(nodes_sum))
            if is_numeric
                # Numeric: left child gets bins [1..threshold]
                for b in 1:bin
                    sum_val += h∇[kk, b, feat, node]
                end
            else
                # Categorical: left child gets only matching bin
                sum_val = h∇[kk, bin, feat, node]
            end
            # Store child sums (used for next level's subtraction)
            nodes_sum[kk, child_l] = sum_val
            nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val  # Already subtraction!
        end
        
        w_l = nodes_sum[2*K_val+1, child_l]
        w_r = nodes_sum[2*K_val+1, child_r]
        
        if K_val == 1
            g_l = nodes_sum[1, child_l]
            h_l = nodes_sum[2, child_l]
            tree_pred[1, child_l] = -g_l / max(eps, h_l + lambda * w_l + L2)
            
            g_r = nodes_sum[1, child_r]
            h_r = nodes_sum[2, child_r]
            tree_pred[1, child_r] = -g_r / max(eps, h_r + lambda * w_r + L2)
        else
            for k in 1:K_val
                g_l = nodes_sum[k, child_l]
                h_l = nodes_sum[K_val+k, child_l]
                tree_pred[k, child_l] = -g_l / max(eps, h_l + lambda * w_l + L2)
                
                g_r = nodes_sum[k, child_r]
                h_r = nodes_sum[K_val+k, child_r]
                tree_pred[k, child_r] = -g_r / max(eps, h_r + lambda * w_r + L2)
            end
        end
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else
        # Make leaf node
        w = nodes_sum[2*K_val+1, node]
        if K_val == 1
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            tree_pred[1, node] = (w <= 0 || h <= 0) ? 0.0f0 : -g / max(eps, h + lambda * w + L2)
        else
            for k in 1:K_val
                g = nodes_sum[k, node]
                h = nodes_sum[K_val+k, node]
                tree_pred[k, node] = (w <= 0 || h <= 0) ? 0.0f0 : -g / max(eps, h + lambda * w + L2)
            end
        end
    end
end

