function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    
    js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
    EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
    copyto!(cache.js, js_cpu)

    tree = EvoTrees.Tree{L,K}(params.max_depth)
    grow! = params.tree_type == "oblivious" ? grow_otree! : grow_tree!
    grow!(
        tree,
        params,
        cache.∇,
        cache.edges,
        cache.nidx,
        is,
        cache.js,
        cache.h∇,
        cache.x_bin,
        cache.feattypes_gpu,
        cache.left_nodes_buf,
        cache.right_nodes_buf,
        cache.target_mask_buf,
    )
    push!(evotree.trees, tree)
    EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    cache[:info][:nrounds] += 1
    return nothing
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes{L},
    ∇::CuMatrix,
    edges,
    nidx::CuVector,
    is::CuVector,
    js::CuVector,
    h∇::CuArray,
    x_bin::CuMatrix,
    feattypes_gpu::CuVector{Bool},
    left_nodes_buf::CuArray{Int32},
    right_nodes_buf::CuArray{Int32},
    target_mask_buf::CuArray{UInt8},
) where {L,K}

    backend = KernelAbstractions.get_backend(x_bin)

    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, length(tree.split))
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt8, length(tree.cond_bin))
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, length(tree.feat))
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, length(tree.gain))
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, size(tree.pred, 1), size(tree.pred, 2))

    max_nodes_total = 2^(params.max_depth + 1)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float32, 3, max_nodes_total)
    nodes_gain_gpu = KernelAbstractions.zeros(backend, Float32, max_nodes_total)

    max_nodes_level = 2^params.max_depth
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = KernelAbstractions.zeros(backend, Int32, 1)

    best_gain_gpu = KernelAbstractions.zeros(backend, Float32, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    
    nidx .= 1
    
    view(anodes_gpu, 1:1) .= 1
    update_hist_gpu!(
        h∇, best_gain_gpu, best_bin_gpu, best_feat_gpu,
        ∇, x_bin, nidx, js, is,
        1, view(anodes_gpu, 1:1), nodes_sum_gpu, params,
        left_nodes_buf, right_nodes_buf, target_mask_buf
    )
    get_gain_gpu!(backend)(nodes_gain_gpu, nodes_sum_gpu, view(anodes_gpu, 1:1), Float32(params.lambda); ndrange=1)

    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        active_nodes_full = view(anodes_gpu, 1:n_nodes_level)
        
        if n_active < n_nodes_level
            view(anodes_gpu, n_active+1:n_nodes_level) .= 0
        end

        view_gain = view(best_gain_gpu, 1:n_nodes_level)
        view_bin  = view(best_bin_gpu, 1:n_nodes_level)
        view_feat = view(best_feat_gpu, 1:n_nodes_level)
        
        if depth > 1
            active_nodes_act = view(active_nodes_full, 1:n_active)  # Define this first

            build_nodes_gpu = KernelAbstractions.zeros(backend, Int32, n_active)
            subtract_nodes_gpu = KernelAbstractions.zeros(backend, Int32, n_active)
            build_count = KernelAbstractions.zeros(backend, Int32, 1)
            subtract_count = KernelAbstractions.zeros(backend, Int32, 1)

            separate_kernel! = separate_nodes_kernel!(backend)
            separate_kernel!(
                build_nodes_gpu, build_count,
                subtract_nodes_gpu, subtract_count,
                active_nodes_act;  # Now it's defined
                ndrange=n_active
            )
            
            # Remove the scalar reads - just launch both kernels unconditionally
            build_nodes_view = view(build_nodes_gpu, 1:n_active)
            update_hist_gpu!(
                h∇, view_gain, view_bin, view_feat,
                ∇, x_bin, nidx, js, is,
                depth, build_nodes_view, nodes_sum_gpu, params,
                left_nodes_buf, right_nodes_buf, target_mask_buf
            )
            
            subtract_nodes_view = view(subtract_nodes_gpu, 1:n_active)
            subtract_kernel! = subtract_hist_kernel!(backend)
            n_work = n_active * size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
            subtract_kernel!(h∇, subtract_nodes_view; ndrange = n_work)
            
            KernelAbstractions.synchronize(backend)
            
            find_split! = find_best_split_from_hist_kernel!(backend)
            find_split!(view_gain, view_bin, view_feat, h∇, nodes_sum_gpu, active_nodes_act, js,
                      Float32(params.lambda), Float32(params.min_weight); ndrange = n_active)
        end

        n_next_active_gpu .= 0
        view_gain_act  = view(view_gain, 1:n_active)
        view_bin_act   = view(view_bin, 1:n_active)
        view_feat_act  = view(view_feat, 1:n_active)

        active_nodes_act = view(active_nodes_full, 1:n_active)

        apply_splits_kernel!(backend)(
            tree_split_gpu, tree_cond_bin_gpu, tree_feat_gpu, tree_gain_gpu, tree_pred_gpu,
            nodes_sum_gpu, nodes_gain_gpu,
            n_next_gpu, n_next_active_gpu,
            view_gain_act, view_bin_act, view_feat_act,
            h∇,
            active_nodes_act,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma);
            ndrange = n_active
        )
        
        n_active = min(2 * n_active, 2^depth)
        if n_active > 0
            copyto!(view(anodes_gpu, 1:n_active), view(n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                nidx, is, x_bin, tree_feat_gpu, tree_cond_bin_gpu, feattypes_gpu;
                ndrange = length(is)
            )
        end
    end

    copyto!(tree.split, Array(tree_split_gpu))
    copyto!(tree.cond_bin, Array(tree_cond_bin_gpu))
    copyto!(tree.feat, Array(tree_feat_gpu))
    copyto!(tree.gain, Array(tree_gain_gpu))
    copyto!(tree.pred, Array(tree_pred_gpu .* Float32(params.eta)))
    
    for i in eachindex(tree.split)
        if tree.split[i]
            tree.cond_float[i] = edges[tree.feat[i]][tree.cond_bin[i]]
        end
    end

    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇,
    active_nodes,
    depth, max_depth, lambda, gamma
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

    epsv = eltype(tree_pred)(1e-8)

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l, child_r = node << 1, (node << 1) + 1
        feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])

        s1 = zero(eltype(nodes_sum)); s2 = zero(eltype(nodes_sum)); s3 = zero(eltype(nodes_sum))
        @inbounds for b in 1:bin
            s1 += h∇[1, b, feat, node]
            s2 += h∇[2, b, feat, node]
            s3 += h∇[3, b, feat, node]
        end
        nodes_sum[1, child_l] = s1
        nodes_sum[2, child_l] = s2
        nodes_sum[3, child_l] = s3
        
        nodes_sum[1, child_r] = nodes_sum[1, node] - nodes_sum[1, child_l]
        nodes_sum[2, child_r] = nodes_sum[2, node] - nodes_sum[2, child_l]
        nodes_sum[3, child_r] = nodes_sum[3, node] - nodes_sum[3, child_l]

        p1_l, p2_l, w_l = nodes_sum[1, child_l], nodes_sum[2, child_l], nodes_sum[3, child_l]
        nodes_gain[child_l] = p1_l^2 / (p2_l + lambda * w_l + epsv)
        p1_r, p2_r, w_r = nodes_sum[1, child_r], nodes_sum[2, child_r], nodes_sum[3, child_r]
        nodes_gain[child_r] = p1_r^2 / (p2_r + lambda * w_r + epsv)
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r

        tree_pred[1, child_l] = - (nodes_sum[1, child_l]) / (nodes_sum[2, child_l] + lambda * nodes_sum[3, child_l] + epsv)
        tree_pred[1, child_r] = - (nodes_sum[1, child_r]) / (nodes_sum[2, child_r] + lambda * nodes_sum[3, child_r] + epsv)
    else
        g, h, w = nodes_sum[1, node], nodes_sum[2, node], nodes_sum[3, node]
        if w <= zero(w) || h + lambda * w <= zero(h)
            tree_pred[1, node] = 0.0f0
        else
            tree_pred[1, node] = -g / (h + lambda * w + epsv)
        end
    end
end

@kernel function get_gain_gpu!(nodes_gain::AbstractVector{T}, nodes_sum::AbstractArray{T,2}, nodes, lambda::T) where {T}
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds p1 = nodes_sum[1, node]
    @inbounds p2 = nodes_sum[2, node]
    @inbounds w = nodes_sum[3, node]
    @inbounds nodes_gain[node] = p1^2 / (p2 + lambda * w + T(1e-8))
end