function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    EvoTrees.sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

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
        cache.h∇L,
        cache.h∇R,
        cache.x_bin,
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
    is,
    js,
    h∇::CuArray,
    h∇L::CuArray,
    h∇R::CuArray,
    x_bin::CuMatrix,
) where {L,K}

    backend = KernelAbstractions.get_backend(x_bin)
    js_gpu = KernelAbstractions.adapt(backend, js)
    is_gpu = KernelAbstractions.adapt(backend, is)

    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, length(tree.split))
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt32, length(tree.cond_bin))
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, length(tree.feat))
    tree_gain_gpu = KernelAbstractions.zeros(backend, K, length(tree.gain))
    tree_pred_gpu = KernelAbstractions.zeros(backend, K, length(tree.pred))

    max_nodes_total = 2^(params.max_depth + 1)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, K, 3, max_nodes_total)
    nodes_gain_gpu = KernelAbstractions.zeros(backend, K, max_nodes_total)

    max_nodes_level = 2^params.max_depth
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = CuArray([0])

    best_gain_gpu = KernelAbstractions.zeros(backend, K, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    
    nidx .= 1
    CUDA.mapreducedim!(x -> (x[1], x[2], K(1.0)), +, view(nodes_sum_gpu, :, 1:1), view(∇, :, is_gpu), dims=2)
    get_gain_gpu!(backend)(nodes_gain_gpu, nodes_sum_gpu, [1], params.lambda; ndrange=1)
    anodes_gpu[1] = 1
    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        active_nodes = view(anodes_gpu, 1:n_active)
        view_gain = view(best_gain_gpu, 1:n_active)
        view_bin = view(best_bin_gpu, 1:n_active)
        view_feat = view(best_feat_gpu, 1:n_active)
        
        update_hist_gpu!(
            h∇, h∇L, h∇R,
            view_gain, view_bin, view_feat,
            ∇, x_bin, nidx, js_gpu,
            depth, active_nodes, params
        )

        n_next_active_gpu .= 0
        apply_splits_kernel!(backend)(
            tree_split_gpu, tree_cond_bin_gpu, tree_feat_gpu, tree_gain_gpu, tree_pred_gpu,
            nodes_sum_gpu, nodes_gain_gpu,
            n_next_gpu, n_next_active_gpu,
            view_gain, view_bin, view_feat,
            h∇L,
            active_nodes,
            depth, params.max_depth, params.lambda, params.gamma;
            ndrange = n_active
        )
        
        n_active = Int(Array(n_next_active_gpu)[1])
        if n_active > 0
            copyto!(view(anodes_gpu, 1:n_active), view(n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                nidx, is_gpu, x_bin, tree_feat_gpu, tree_cond_bin_gpu, params.feattypes_gpu;
                ndrange = length(is_gpu)
            )
        end
    end

    copyto!(tree.split, Array(tree_split_gpu))
    copyto!(tree.cond_bin, Array(tree_cond_bin_gpu))
    copyto!(tree.feat, Array(tree_feat_gpu))
    copyto!(tree.gain, Array(tree_gain_gpu))
    copyto!(tree.pred, Array(tree_pred_gpu))
    
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
    h∇L,
    active_nodes,
    depth, max_depth, lambda, gamma
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

    @inbounds if depth < max_depth && best_gain[n_idx] > nodes_gain[node] + gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l, child_r = node << 1, (node << 1) + 1
        feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])

        nodes_sum[1, child_l] = h∇L[1, bin, feat, node]
        nodes_sum[2, child_l] = h∇L[2, bin, feat, node]
        nodes_sum[3, child_l] = h∇L[3, bin, feat, node]
        
        nodes_sum[1, child_r] = nodes_sum[1, node] - nodes_sum[1, child_l]
        nodes_sum[2, child_r] = nodes_sum[2, node] - nodes_sum[2, child_l]
        nodes_sum[3, child_r] = nodes_sum[3, node] - nodes_sum[3, child_l]

        p1_l, p2_l = nodes_sum[1, child_l], nodes_sum[2, child_l]
        nodes_gain[child_l] = p1_l^2 / (p2_l + lambda)
        p1_r, p2_r = nodes_sum[1, child_r], nodes_sum[2, child_r]
        nodes_gain[child_r] = p1_r^2 / (p2_r + lambda)
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else
        g, h, w = nodes_sum[1, node], nodes_sum[2, node], nodes_sum[3, node]
        tree_pred[node] = -g / (h + lambda)
    end
end

@kernel function get_gain_gpu!(nodes_gain, nodes_sum, nodes, lambda)
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds p1 = nodes_sum[1, node]
    @inbounds p2 = nodes_sum[2, node]
    @inbounds nodes_gain[node] = p1^2 / (p2 + lambda)
end

function grow_otree!(
    tree::EvoTrees.Tree{L,K}, nodes::Vector{N}, params::EvoTrees.EvoTypes{L}, ∇::CuMatrix, edges, js,
    out, left, right, h∇_cpu::Array{Float64,3}, h∇::CuArray{Float64,3}, x_bin::CuMatrix,
    feattypes::Vector{Bool}, monotone_constraints
) where {L,K,N}
    backend = KernelAbstractions.get_backend(x_bin)
    jsg = KernelAbstractions.adapt(backend, js)
    for n in nodes
        n.∑ .= 0; n.gain = 0.0
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0; n.gains[i] .= 0
        end
    end
    n_current = [1]; depth = 1
    nodes[1].∑ .= Vector(vec(sum(∇[:, nodes[1].is], dims=2)))
    nodes[1].gain = EvoTrees.get_gain(params, nodes[1].∑)
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0; n_next = Int[]
        min_weight_flag = false
        for n in n_current
            nodes[n].∑[end] <= params.min_weight ? min_weight_flag = true : nothing
        end
        if depth == params.max_depth || min_weight_flag
            for n in n_current
                EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
            end
        else
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0; @inbounds for j in js; nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j] end
                    else; @inbounds for j in js; nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j] end
                    end
                else
                    update_hist_gpu!(nodes[n].h, h∇_cpu, h∇, ∇, x_bin, nodes[n].is, jsg, js)
                end
            end
            Threads.@threads for n ∈ n_current
                EvoTrees.update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
            end
            if depth > 1; @inbounds for j in js; nodes[1].gains[j] .= 0 end; end
            gain = 0
            for n ∈ sort(n_current)
                if n > 1; for j in js; nodes[1].gains[j] .+= nodes[n].gains[j] end; end
                gain += nodes[n].gain
            end
            for n ∈ sort(n_current)
                if n > 1; for j in js; nodes[1].gains[j] .*= nodes[n].gains[j] .> 0 end; end
            end
            best = findmax(findmax.(nodes[1].gains)); best_gain = best[1][1]; best_bin = best[1][2]; best_feat = best[2]
            if best_gain > gain + params.gamma
                for n in sort(n_current)
                    tree.gain[n] = best_gain - nodes[n].gain; tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat; tree.cond_float[n] = edges[best_feat][best_bin]
                    tree.split[n] = best_bin != 0
                    _left, _right = split_set_threads_gpu!(out,left,right,nodes[n].is,x_bin,tree.feat[n],tree.cond_bin[n],feattypes[best_feat],offset)
                    offset += length(nodes[n].is)
                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[best_feat][:, best_bin]
                    nodes[n<<1+1].∑ .= nodes[n].hR[best_feat][:, best_bin]
                    nodes[n<<1].gain = EvoTrees.get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = EvoTrees.get_gain(params, nodes[n<<1+1].∑)
                    if length(_right) >= length(_left); push!(n_next, n << 1); push!(n_next, (n << 1) + 1)
                    else; push!(n_next, (n << 1) + 1); push!(n_next, n << 1)
                    end
                end
            else
                for n in n_current
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
                end
            end
        end
        n_current = copy(n_next); depth += 1
    end
    return nothing
end

