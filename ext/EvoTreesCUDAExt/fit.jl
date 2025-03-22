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
    h∇::CuArray,
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

        # pred leafs if max depth is reached
        if depth == params.max_depth
            for n ∈ n_current
                if L <: Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            end
        else
            # find best split for each node
            for n ∈ n_current[1:2:end]
                update_hist!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg)
            end
            @threads for n ∈ n_current[2:2:end]
                if n % 2 == 0
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n+1].h[:, :, js]
                else
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n-1].h[:, :, js]
                end
            end
            sort!(n_current)
            @threads for n ∈ n_current
                best_gain, best_feat, best_bin = get_best_split(L, nodes[n], js, params, feattypes, monotone_constraints)
                if best_bin != 0
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.split[n] = true
                end
            end

            for n ∈ n_current
                if tree.split[n]

                    best_feat = tree.feat[n]
                    best_bin = tree.cond_bin[n]

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
    h∇::CuArray,
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

        if depth == params.max_depth
            for n in n_current
                if L <: Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            end
        else
            for n ∈ n_current[1:2:end]
                update_hist!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg)
            end
            @threads for n ∈ n_current[2:2:end]
                if n % 2 == 0
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n+1].h[:, :, js]
                else
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n-1].h[:, :, js]
                end
            end
            sort!(n_current)
            @threads for n ∈ n_current
                update_gains!(L, nodes[n], js, params, feattypes, monotone_constraints)
            end

            # initialize gains for node 1 in which all gains of a given depth will be accumulated
            if depth > 1
                view(nodes[1].gains, :, js) .= 0
            end
            gain = 0
            # update gains based on the aggregation of all nodes of a given depth. One gains matrix per depth (vs one per node in binary trees).
            for n ∈ n_current
                if n > 1 # accumulate gains in node 1
                    @views nodes[1].gains[:, js] .+= nodes[n].gains[:, js]
                end
                gain += nodes[n].gain
            end
            for n ∈ n_current
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
                for n in n_current
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
