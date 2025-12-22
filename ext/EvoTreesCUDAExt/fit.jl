function EvoTrees.grow_evotree!(m::EvoTree{L,K}, cache::EvoTrees.CacheGPU, params::EvoTrees.EvoTypes) where {L,K}

    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)

    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_full, cache.mask_cpu, cache.mask_gpu, params.rowsample, cache.rng)

        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(cache.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)

        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow_tree!(tree, params, cache, is)
        push!(m.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end

    m.info[:nrounds] += 1
    return nothing
end

"""
	grow_otree!(tree, params, cache, is)

Grow an oblivious tree on GPU.

GPU oblivious-tree growth is not implemented yet; falls back to `grow_tree!` and emits a warning.

Mutates:
- `tree`: the resulting tree structure and leaf predictions
- `cache`: internal GPU working buffers used during growth
"""
function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::EvoTrees.CacheGPU,
    is::CuVector,
) where {L,K}
    @warn "Oblivious tree GPU implementation not yet available, using standard tree" maxlog = 1
    grow_tree!(tree, params, cache, is)
end

"""
	grow_tree!(tree, params, cache, is)

Grow a binary decision tree on GPU, level-by-level (breadth-first).

Mutates:
- `tree`: split structure and leaf predictions (copied back from GPU buffers)
- `cache`: GPU working buffers (histograms, node lists, gains, etc.)
"""
function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::EvoTrees.CacheGPU,
    is::CuVector,
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

    n_feats = length(cache.js)

    # Root node processing
    if params.max_depth == 1
        reduce_root_sums_kernel!(backend)(
            cache.nodes_sum_gpu, ∇_gpu, is;
            ndrange=length(is),
        )
        KernelAbstractions.synchronize(backend)
        n_active = 0
    else
        update_hist_gpu!(
            cache.h∇, ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
            view(cache.anodes_gpu, 1:1), cache.K, cache.target_mask_buf, backend,
        )

        compute_nodes_sum_kernel!(backend)(
            cache.nodes_sum_gpu, cache.h∇, view(cache.anodes_gpu, 1:1), cache.K;
            ndrange=(2 * cache.K + 1),
        )
        KernelAbstractions.synchronize(backend)

        find_best_split_parallel_kernel!(backend)(
            L,
            view(cache.gains_per_feat_gpu, 1:n_feats, 1:1),
            view(cache.bins_per_feat_gpu, 1:n_feats, 1:1),
            cache.h∇, cache.nodes_sum_gpu,
            view(cache.anodes_gpu, 1:1),
            cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
            params.lambda, params.L2, params.min_weight,
            cache.K, n_feats, cache.split_sums_temp_gpu;
            ndrange=n_feats,
        )
        KernelAbstractions.synchronize(backend)

        reduce_best_split_kernel!(backend)(
            view(cache.best_gain_gpu, 1:1),
            view(cache.best_bin_gpu, 1:1),
            view(cache.best_feat_gpu, 1:1),
            view(cache.gains_per_feat_gpu, 1:n_feats, 1:1),
            view(cache.bins_per_feat_gpu, 1:n_feats, 1:1),
            cache.js, n_feats;
            ndrange=1,
        )
        KernelAbstractions.synchronize(backend)

        n_active = 1
    end

    # Main loop: build tree level by level
    for depth in 1:(params.max_depth-1)
        iszero(n_active) && break

        view(cache.n_next_active_gpu, 1:1) .= 0
        active_nodes = view(cache.anodes_gpu, 1:n_active)

        # Histogram subtraction (depth >= 2)
        if depth >= 2
            cache.build_nodes_gpu .= 0
            cache.subtract_nodes_gpu .= 0
            cache.build_count .= 0
            cache.subtract_count .= 0

            cache.node_counts_gpu .= 0
            count_nodes_kernel!(backend)(
                cache.node_counts_gpu, cache.nidx, is;
                ndrange=length(is)
            )
            KernelAbstractions.synchronize(backend)

            separate_nodes_kernel!(backend)(
                cache.build_nodes_gpu, cache.build_count,
                cache.subtract_nodes_gpu, cache.subtract_count,
                active_nodes, cache.node_counts_gpu;
                ndrange=n_active
            )
            KernelAbstractions.synchronize(backend)

            build_count_val = Array(cache.build_count)[1]
            subtract_count_val = Array(cache.subtract_count)[1]

            # Build histograms for smaller children
            if build_count_val > 0
                update_hist_gpu!(
                    cache.h∇, ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
                    view(cache.build_nodes_gpu, 1:build_count_val),
                    cache.K, cache.target_mask_buf, backend,
                )
            end

            # Compute larger children via subtraction
            if subtract_count_val > 0
                subtract_hist_kernel!(backend)(
                    cache.h∇,
                    view(cache.subtract_nodes_gpu, 1:subtract_count_val);
                    ndrange=subtract_count_val * size(cache.h∇, 1) * size(cache.h∇, 2) * size(cache.h∇, 3),
                )
                KernelAbstractions.synchronize(backend)
            end

            compute_nodes_sum_kernel!(backend)(
                cache.nodes_sum_gpu, cache.h∇, active_nodes, cache.K;
                ndrange=n_active * (2 * cache.K + 1),
            )
            KernelAbstractions.synchronize(backend)

            find_best_split_parallel_kernel!(backend)(
                L,
                view(cache.gains_per_feat_gpu, 1:n_feats, 1:n_active),
                view(cache.bins_per_feat_gpu, 1:n_feats, 1:n_active),
                cache.h∇, cache.nodes_sum_gpu,
                active_nodes,
                cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
                params.lambda, params.L2, params.min_weight,
                cache.K, n_feats, cache.split_sums_temp_gpu;
                ndrange=n_active * n_feats,
            )
            KernelAbstractions.synchronize(backend)

            reduce_best_split_kernel!(backend)(
                view(cache.best_gain_gpu, 1:n_active),
                view(cache.best_bin_gpu, 1:n_active),
                view(cache.best_feat_gpu, 1:n_active),
                view(cache.gains_per_feat_gpu, 1:n_feats, 1:n_active),
                view(cache.bins_per_feat_gpu, 1:n_feats, 1:n_active),
                cache.js, n_feats;
                ndrange=n_active,
            )
            KernelAbstractions.synchronize(backend)
        end

        # Apply splits
        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu,
            cache.tree_gain_gpu, cache.nodes_sum_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view(cache.best_gain_gpu, 1:n_active),
            view(cache.best_bin_gpu, 1:n_active),
            view(cache.best_feat_gpu, 1:n_active),
            cache.h∇, active_nodes, cache.feattypes_gpu,
            depth, params.max_depth, Float32(params.gamma),
            cache.K;
            ndrange=max(n_active, 1),
        )
        KernelAbstractions.synchronize(backend)

        n_active = Array(cache.n_next_active_gpu)[1]
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        # Update observation->node assignments
        if n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu,
                cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange=length(is),
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Copy tree to CPU and compute leaf predictions
    copyto!(tree.split, cache.tree_split_gpu)
    copyto!(tree.feat, cache.tree_feat_gpu)
    copyto!(tree.cond_bin, cache.tree_cond_bin_gpu)
    copyto!(tree.gain, cache.tree_gain_gpu)
    copyto!(tree.w, view(cache.nodes_sum_gpu, size(cache.nodes_sum_gpu, 1), 1:length(tree.w)))

    leaf_nodes = findall(!, tree.split)

    if L <: EvoTrees.Quantile
        cpu_data = (
            nidx=Array(cache.nidx),
            is=Array(is),
            ∇=Array(cache.∇),
            nodes_sum=Array(cache.nodes_sum_gpu),
        )

        leaf_map = Dict{Int,Vector{UInt32}}()
        sizehint!(leaf_map, length(leaf_nodes))
        for i in 1:length(cpu_data.is)
            leaf_id = cpu_data.nidx[cpu_data.is[i]]
            if leaf_id > 0 && leaf_id <= length(tree.split) && !tree.split[leaf_id]
                if !haskey(leaf_map, leaf_id)
                    leaf_map[leaf_id] = UInt32[]
                end
                push!(leaf_map[leaf_id], cpu_data.is[i])
            end
        end

        Threads.@threads for n in leaf_nodes
            node_sum_view = view(cpu_data.nodes_sum, :, n)
            node_is = get(leaf_map, n, UInt32[])
            if !isempty(node_is)
                EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params, cpu_data.∇, node_is)
            else
                tree.pred[:, n] .= 0
            end
        end
    else
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        Threads.@threads for n in leaf_nodes
            node_sum_view = view(nodes_sum_cpu, :, n)
            EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params)
        end
    end

    return nothing
end

"""
	apply_splits_kernel!(tree_split, tree_cond_bin, tree_feat, tree_gain,
	                     nodes_sum, n_next, n_next_active,
	                     best_gain, best_bin, best_feat,
	                     h∇, active_nodes, feattypes,
	                     depth, max_depth, gamma, K_val)

Apply the chosen best split for each active node and create its children.

For each active node `node = active_nodes[n_idx]`, if `best_gain[n_idx] > gamma`
and we are below `max_depth`, mark the node as split and:
- Write split metadata (`tree_split`, `tree_feat`, `tree_cond_bin`, `tree_gain`)
- Compute left-child gradient totals from histograms (`h∇`) and write them into `nodes_sum`
- Compute right-child totals as `parent - left` (also into `nodes_sum`)
- Append the two children to the next active-node list (`n_next`) using atomic allocation

Mutates:
- `tree_split`, `tree_cond_bin`, `tree_feat`, `tree_gain`
- `nodes_sum` (writes child node totals)
- `n_next`, `n_next_active`
"""
@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain,
    nodes_sum, n_next, n_next_active,
    best_gain, best_bin, best_feat, h∇, active_nodes, feattypes,
    depth, max_depth, gamma, K_val,
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

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

        for kk in 1:(2*K_val+1)
            sum_val = zero(eltype(nodes_sum))
            if is_numeric
                for b in 1:bin
                    sum_val += h∇[kk, b, feat, node]
                end
            else
                sum_val = h∇[kk, bin, feat, node]
            end
            nodes_sum[kk, child_l] = sum_val
            nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val
        end

        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base-1] = child_l
        n_next[idx_base] = child_r
    end
end

