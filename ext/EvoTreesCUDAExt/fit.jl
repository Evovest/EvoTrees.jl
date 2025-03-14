function EvoTrees.grow_evotree!(m::EvoTree{L,K}, cache::EvoTrees.CacheGPU, params::EvoTrees.EvoTypes) where {L,K}

    # compute gradients
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    # subsample rows
    cache.nodes[1].is =
        EvoTrees.subsample(cache.left, cache.is, cache.mask_cond, params.rowsample, params.rng)
    # subsample cols
    EvoTrees.sample!(params.rng, UInt32(1):UInt32(length(cache.feattypes)), cache.js, replace=false, ordered=true)

    # assign a root and grow tree
    tree = EvoTrees.Tree{L,K}(params.max_depth)
    grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
    grow!(
        tree,
        cache.nodes,
        params,
        cache.∇,
        cache.js,
        cache.is,
        cache.left,
        cache.right,
        cache.h∇_cpu,
        cache.h∇,
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints,
    )
    push!(m.trees, tree)
    EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    m.info[:nrounds] += 1
    return nothing
end

# grow a single binary tree - grow through all depth
function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    nodes::Vector{N},
    params::EvoTrees.EvoTypes,
    ∇::CuMatrix,
    js,
    is,
    left,
    right,
    h∇_cpu::Array{Float64,3},
    h∇::CuArray{Float64,3},
    x_bin::CuMatrix,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L,K,N}

    # initialize
    n_current = [1]
    depth = 1
    jsg = CuVector(js)

    # initialize summary stats
    copyto!(nodes[1].∑, sum(Float64, view(∇, :, nodes[1].is), dims=2))
    nodes[1].gain = get_gain(L, params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        # update histograms and gains
        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n+1].h[:, :, js]
                    else
                        @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n-1].h[:, :, js]
                    end
                else
                    update_hist!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg)
                end
            end
            @threads for n ∈ n_current
                update_gains!(L, nodes[n], js, params, feattypes, monotone_constraints)
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                if L <: Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            else
                best = findmax(view(nodes[n].gains, :, js))
                best_gain = best[1]
                best_bin = best[2][1]
                best_feat = js[best[2][2]]
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.split[n] = best_bin != 0

                    _left, _right = split_set!(
                        nodes[n].is,
                        is,
                        left,
                        right,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )
                    offset += length(nodes[n].is)

                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[:, best_bin, best_feat]
                    nodes[n<<1+1].∑ .= nodes[n].hR[:, best_bin, best_feat]
                    nodes[n<<1].gain = get_gain(L, params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(L, params, nodes[n<<1+1].∑)

                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
                else
                    if L <: Quantile
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                    else
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                    end
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth

    return nothing
end


# grow a single oblivious tree - grow through all depth
function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    nodes::Vector{N},
    params::EvoTrees.EvoTypes,
    ∇::CuMatrix,
    js,
    is,
    left,
    right,
    h∇_cpu::Array{Float64,3},
    h∇::CuArray{Float64,3},
    x_bin::CuMatrix,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L,K,N}

    # initialize
    n_current = [1]
    depth = 1
    jsg = CuVector(js)

    # initialize summary stats
    copyto!(nodes[1].∑, sum(Float64, view(∇, :, nodes[1].is), dims=2))
    nodes[1].gain = get_gain(L, params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        min_weight_flag = false
        for n in n_current
            nodes[n].∑[end] <= params.min_weight ? min_weight_flag = true : nothing
        end
        if depth == params.max_depth || min_weight_flag
            for n in n_current
                if L <: EvoTrees.Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            end
        else
            # update histograms
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n+1].h[:, :, js]
                    else
                        @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n-1].h[:, :, js]
                    end
                else
                    update_hist!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg)
                end
            end
            @threads for n ∈ n_current
                update_gains!(L, nodes[n], js, params, feattypes, monotone_constraints)
            end

            # initialize gains for node 1 in which all gains of a given depth will be accumulated
            if depth > 1
                view(nodes[1].gains, :, js) .= 0
            end
            gain = 0
            # update gains based on the aggregation of all nodes of a given depth. One gains matrix per depth (vs one per node in binary trees).
            for n ∈ sort(n_current)
                if n > 1 # accumulate gains in node 1
                    @views nodes[1].gains[:, js] .+= nodes[n].gains[:, js]
                end
                gain += nodes[n].gain
            end
            for n ∈ sort(n_current)
                if n > 1
                    @views nodes[1].gains[:, js] .*= nodes[n].gains[:, js] .> 0 #mask ignore gains if any node isn't eligible (too small per leaf weight)
                end
            end
            # find best split
            best = findmax(view(nodes[1].gains, :, js))
            best_gain = best[1]
            best_bin = best[2][1]
            best_feat = js[best[2][2]]
            if best_gain > gain + params.gamma
                for n in sort(n_current)
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.split[n] = best_bin != 0

                    _left, _right = split_set!(
                        nodes[n].is,
                        is,
                        left,
                        right,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )
                    offset += length(nodes[n].is)

                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[:, best_bin, best_feat]
                    nodes[n<<1+1].∑ .= nodes[n].hR[:, best_bin, best_feat]
                    nodes[n<<1].gain = get_gain(L, params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(L, params, nodes[n<<1+1].∑)

                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
                end
            else
                for n in n_current
                    if L <: Quantile
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                    else
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                    end
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over current nodes for a given depth

    return nothing
end
