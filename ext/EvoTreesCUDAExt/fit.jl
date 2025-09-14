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
    
    if isdefined(cache, :node_counts)
        cache.node_counts .= 0
    end
    if isdefined(cache, :build_mask)
        cache.build_mask .= 0
    end
    
    view(cache.anodes_gpu, 1:1) .= 1

    update_hist_gpu!(
        cache.h∇, cache.h∇_parent, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.∇, cache.x_bin, cache.nidx, cache.js, is,
        1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
        cache.node_counts, cache.build_mask,  
        cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K;
        is_mae=(L <: EvoTrees.MAE), is_quantile=(L <: EvoTrees.Quantile), 
        is_cred=(L <: EvoTrees.Cred), is_mle2p=(L <: EvoTrees.MLE2P)
    )
    
    get_gain_gpu!(backend)(
        cache.nodes_gain_gpu, cache.nodes_sum_gpu, view(cache.anodes_gpu, 1:1), 
        Float32(params.lambda), Float32(params.L2), cache.K, (L <: EvoTrees.MLE2P); 
        ndrange=1, workgroupsize=1
    )
    KernelAbstractions.synchronize(backend)

    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        
        if depth > 1
            
            active_nodes_act = view(cache.anodes_gpu, 1:n_active)
            
            update_hist_gpu!(
                cache.h∇, cache.h∇_parent, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                cache.∇, cache.x_bin, cache.nidx, cache.js, is,
                depth, active_nodes_act, cache.nodes_sum_gpu, params,
                cache.node_counts, cache.build_mask,  
                cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K;
                is_mae=(L <: EvoTrees.MAE), is_quantile=(L <: EvoTrees.Quantile), 
                is_cred=(L <: EvoTrees.Cred), is_mle2p=(L <: EvoTrees.MLE2P)
            )
        end
        
        is_mae = L <: EvoTrees.MAE
        is_quantile = L <: EvoTrees.Quantile
        is_cred = L <: EvoTrees.Cred
        is_mle2p = L <: EvoTrees.MLE2P
        alpha = is_quantile ? Float32(params.alpha) : 0.5f0

        if is_mae || is_quantile
            nidx_host = Array(cache.nidx)
            y_host = Array(cache.y)
            pred_host = Array(view(cache.pred, 1, :))
            
            max_node = Int(2^(depth) - 1)
            leaf_vals = zeros(Float32, max_node)
            
            for node in 1:max_node
                idxs = findall(==(node), nidx_host)
                if !isempty(idxs)
                    residuals = y_host[idxs] .- pred_host[idxs]
                    if is_mae
                        leaf_vals[node] = median(residuals) / Float32(params.bagging_size)
                    else 
                        leaf_vals[node] = quantile(residuals, alpha) / Float32(params.bagging_size)
                    end
                end
            end
            
            copyto!(view(cache.nodes_sum_gpu, 1, 1:max_node), leaf_vals)
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, 
            cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu, cache.nodes_gain_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view(cache.best_gain_gpu, 1:n_nodes_level), 
            view(cache.best_bin_gpu, 1:n_nodes_level), 
            view(cache.best_feat_gpu, 1:n_nodes_level),
            cache.h∇,
            view(cache.anodes_gpu, 1:n_active),
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), Float32(params.L2),
            K, is_quantile, is_mae, is_cred, is_mle2p, alpha, Float32(params.bagging_size);
            ndrange = n_active, workgroupsize=min(256, n_active)
        )
        
        n_active_h = Array(cache.n_next_active_gpu)
        n_active = n_active_h[1]
        cache.n_next_active_gpu .= 0
        
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, 
                cache.tree_cond_bin_gpu, cache.feattypes_gpu;
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
    K, is_quantile::Bool, is_mae::Bool, is_cred::Bool, is_mle2p::Bool, alpha::Float32, bagging_size::Float32
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
        elseif is_cred
            w = nodes_sum[2*K+1, node]
            if w > epsv
                g = nodes_sum[1, node]
                tree_pred[1, node] = g / (w + L2 + epsv) / bagging_size
            end
        elseif is_mle2p
            w = nodes_sum[5, node]
            if w > epsv
                g1 = nodes_sum[1, node]
                h1 = nodes_sum[3, node]
                g2 = nodes_sum[2, node]
                h2 = nodes_sum[4, node]
                tree_pred[1, node] = -g1 / (h1 + lambda * w + L2 + epsv) / bagging_size
                tree_pred[2, node] = -g2 / (h2 + lambda * w + L2 + epsv) / bagging_size
            end
        else
            w = nodes_sum[2*K+1, node]
            if w > epsv
                @inbounds for kk in 1:K
                    gk = nodes_sum[kk, node]
                    hk = nodes_sum[K+kk, node]
                    tree_pred[kk, node] = -gk / (hk + lambda * w + L2 + epsv) / bagging_size
                end
            end
        end
    end
end

@kernel function get_gain_gpu!(nodes_gain::AbstractVector{T}, nodes_sum::AbstractArray{T,2}, nodes, lambda::T, L2::T, K::Int, is_mle2p::Bool=false) where {T}
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds if node > 0
        eps = T(1e-8)
        gain = zero(T)
        
        if is_mle2p && K == 2
            w = nodes_sum[5, node]
            if w > eps
                g1, g2 = nodes_sum[1, node], nodes_sum[2, node]
                h1, h2 = nodes_sum[3, node], nodes_sum[4, node]
                gain = (g1^2 / max(eps, (h1 + lambda * w + L2)) + g2^2 / max(eps, (h2 + lambda * w + L2))) / 2
            end
        else
            w = nodes_sum[2*K+1, node]
            if w > eps
                @inbounds for kk in 1:K
                    g = nodes_sum[kk, node]
                    h = nodes_sum[K+kk, node]
                    gain += g^2 / max(eps, (h + lambda * w + L2))
                end
                gain /= 2
            end
        end
        
        nodes_gain[node] = gain
    end
end
