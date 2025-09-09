function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
        grow!(
            tree,
            params,
            cache,
            is,
        )
        push!(evotree.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end
    
    evotree.info[:nrounds] += 1
    return nothing
end

function grow_otree!(tree::EvoTrees.Tree{L,K}, params::EvoTrees.EvoTypes, cache::CacheGPU, is::CuVector) where {L,K}
    @warn "Oblivious tree GPU implementation not yet available, using standard tree" maxlog=1
    grow_tree!(tree, params, cache, is)
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}

    backend = get_backend(cache.x_bin)

    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0.0
    cache.tree_pred_gpu .= 0.0
    cache.nodes_sum_gpu .= 0.0
    cache.nodes_gain_gpu .= 0.0
    cache.anodes_gpu .= 0
    cache.n_next_gpu .= 0
    cache.n_next_active_gpu .= 0
    cache.best_gain_gpu .= -Inf
    cache.best_bin_gpu .= 0
    cache.best_feat_gpu .= 0
    cache.nidx .= 1
    
    view(cache.anodes_gpu, 1:1) .= 1

    update_hist_gpu!(
        cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.∇, cache.x_bin, cache.nidx, cache.js, is,
        1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
        cache.left_nodes_buf, cache.right_nodes_buf, cache.target_mask_buf, 
        cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K
    )
    
    get_gain_gpu!(backend)(cache.nodes_gain_gpu, cache.nodes_sum_gpu, view(cache.anodes_gpu, 1:1), Float32(params.lambda), Float32(params.L2), cache.K; ndrange=1, workgroupsize=1)
    KernelAbstractions.synchronize(backend)

    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        
        if depth > 1
            active_nodes_act = view(cache.anodes_gpu, 1:n_active)

            cache.build_nodes_gpu .= 0
            cache.subtract_nodes_gpu .= 0
            cache.build_count .= 0
            cache.subtract_count .= 0

            separate_nodes_kernel!(backend)(
                cache.build_nodes_gpu, cache.build_count,
                cache.subtract_nodes_gpu, cache.subtract_count,
                active_nodes_act;
                ndrange=n_active, workgroupsize=min(256, n_active)
            )
            KernelAbstractions.synchronize(backend)
            
            subtract_count_val = Array(cache.subtract_count)[1]
            build_count_val = Array(cache.build_count)[1]
            
            if subtract_count_val > 0
                n_k, n_b, n_j = size(cache.h∇, 1), size(cache.h∇, 2), size(cache.h∇, 3)
                subtract_hist_kernel!(backend)(
                    cache.h∇, view(cache.subtract_nodes_gpu, 1:subtract_count_val), n_k, n_b, n_j;
                    ndrange = subtract_count_val * n_k * n_b * n_j, workgroupsize=256
                )
                KernelAbstractions.synchronize(backend)
            end
            
            if build_count_val > 0
                
                update_hist_gpu!(
                    cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                    cache.∇, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), cache.nodes_sum_gpu, params,
                    cache.left_nodes_buf, cache.right_nodes_buf, cache.target_mask_buf, 
                    cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K
                )
            end
        end
        
        is_mae = L <: EvoTrees.MAE
        is_quantile = L <: EvoTrees.Quantile
        alpha = is_mae ? 0.5f0 : Float32(params.alpha)

        if is_mae || is_quantile
            
            nidx_host = Array(cache.nidx)
            res_host = Array(view(cache.∇, 2, :)) 
            w_host = Array(view(cache.∇, size(cache.∇, 1), :)) 
            max_node = maximum(nidx_host)
            pre_leaf = zeros(Float32, max_node)
            for node in 1:max_node
                idxs = findall(==(node), nidx_host)
                if !isempty(idxs)
                    if is_mae
                        
                        g = sum(res_host[idxs] .* w_host[idxs])
                        wsum = sum(w_host[idxs])
                        denom = wsum + Float32(params.lambda) * wsum + Float32(params.L2)
                        pre = denom > 0 ? g / denom : 0f0
                        pre_leaf[node] = pre / Float32(params.bagging_size)
                    else
                        
                        vals = res_host[idxs]
                        wsum = sum(w_host[idxs])
                        denom = 1f0 + Float32(params.lambda) + (wsum > 0 ? Float32(params.L2) / wsum : Float32(params.L2))
                        q = quantile(vals, alpha)
                        pre_leaf[node] = (Float32(q) / denom) / Float32(params.bagging_size)
                    end
                end
            end
            copyto!(cache.pre_leaf_gpu, pre_leaf)
            
            cache.nodes_sum_gpu[1, 1:length(pre_leaf)] .= cache.pre_leaf_gpu[1:length(pre_leaf)]
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu, cache.nodes_gain_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view(cache.best_gain_gpu, 1:n_nodes_level), 
            view(cache.best_bin_gpu, 1:n_nodes_level), 
            view(cache.best_feat_gpu, 1:n_nodes_level),
            cache.h∇,
            view(cache.anodes_gpu, 1:n_nodes_level),
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), Float32(params.L2),
            K, is_quantile, is_mae, alpha;
            ndrange = n_active, workgroupsize=min(256, n_active)
        )
        
        n_active_h = Array(cache.n_next_active_gpu)
        n_active = n_active_h[1]
        
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))
    copyto!(tree.pred, Array(cache.tree_pred_gpu .* Float32(params.eta)))
    
    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇,
    active_nodes,
    depth, max_depth, lambda, gamma, L2,
    K, is_quantile::Bool, is_mae::Bool, alpha::Float32
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]
    epsv = eltype(tree_pred)(1e-8)

    @inbounds if node > 0 && depth < max_depth && best_gain[n_idx] > gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l, child_r = node << 1, (node << 1) + 1
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else 
        if is_mae || is_quantile
            
            tree_pred[1, node] = nodes_sum[1, node]
        else
            w = nodes_sum[2*K+1, node]
            if w > epsv
                @inbounds for kk in 1:K
                    gk = nodes_sum[kk, node]
                    hk = nodes_sum[K+kk, node]
                    tree_pred[kk, node] = -gk / (hk + lambda * w + L2 + epsv)
                end
            end
        end
    end
end

@kernel function get_gain_gpu!(nodes_gain::AbstractVector{T}, nodes_sum::AbstractArray{T,2}, nodes, lambda::T, L2::T, K::Int) where {T}
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds if node > 0
        w = nodes_sum[2*K+1, node]
        gain = zero(T)
        if w > 1.0f-8
            @inbounds for kk in 1:K
                p1 = nodes_sum[kk, node]
                p2 = nodes_sum[K+kk, node]
                gain += p1^2 / (p2 + lambda * w + L2)
            end
        end
        nodes_gain[node] = gain
    end
end
