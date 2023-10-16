function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}

    # compute gradients
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    cache.nodes[1].is =
        EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    # subsample cols
    EvoTrees.sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    # assign a root and grow tree
    tree = EvoTrees.Tree{L,K}(params.max_depth)
    grow! = params.tree_type == "oblivious" ? grow_otree! : grow_tree!
    grow!(
        tree,
        cache.nodes,
        params,
        cache.∇,
        cache.edges,
        cache.js,
        cache.out,
        cache.left,
        cache.right,
        cache.h∇_cpu,
        cache.h∇,
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints,
    )
    push!(evotree.trees, tree)
    EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single binary tree - grow through all depth
function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    nodes::Vector{N},
    params::EvoTrees.EvoTypes{L},
    ∇::CuMatrix,
    edges,
    js,
    out,
    left,
    right,
    h∇_cpu::Array{Float64,3},
    h∇::CuArray{Float64,3},
    x_bin::CuMatrix,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L,K,N}

    jsg = CuVector(js)
    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = 0.0
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= Vector(vec(sum(∇[:, nodes[1].is], dims=2)))
    nodes[1].gain = EvoTrees.get_gain(params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        if depth < params.max_depth

            update_hist_gpu_single!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg, js)
            hist_gpu_to_cpu_copy!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg, js)
            # for n_id in eachindex(n_current)
            #     n = n_current[n_id]
            #     if n_id % 2 == 0
            #         if n % 2 == 0
            #             @inbounds for j in js
            #                 nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
            #             end
            #         else
            #             @inbounds for j in js
            #                 nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
            #             end
            #         end
            #     else
            #         update_hist_gpu!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg, js)
            #     end
            # end
            Threads.@threads for n ∈ sort(n_current)
                EvoTrees.update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
            else
                best = findmax(findmax.(nodes[n].gains))
                best_gain = best[1][1]
                best_bin = best[1][2]
                best_feat = best[2]
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                    tree.split[n] = best_bin != 0

                    # _left, _right = split_set_threads_gpu!(
                    #     out,
                    #     left,
                    #     right,
                    #     nodes[n].is,
                    #     x_bin,
                    #     tree.feat[n],
                    #     tree.cond_bin[n],
                    #     feattypes[best_feat],
                    #     offset,
                    # )

                    offset += length(nodes[n].is)
                    # nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[best_feat][:, best_bin]
                    nodes[n<<1+1].∑ .= nodes[n].hR[best_feat][:, best_bin]
                    nodes[n<<1].gain = EvoTrees.get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = EvoTrees.get_gain(params, nodes[n<<1+1].∑)

                    # if length(_right) >= length(_left)
                    #     push!(n_next, n << 1)
                    #     push!(n_next, n << 1 + 1)
                    # else
                    push!(n_next, n << 1 + 1)
                    push!(n_next, n << 1)
                    # end
                else
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
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
    params::EvoTrees.EvoTypes{L},
    ∇::CuMatrix,
    edges,
    js,
    out,
    left,
    right,
    h∇_cpu::Array{Float64,3},
    h∇::CuArray{Float64,3},
    x_bin::CuMatrix,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L,K,N}

    jsg = CuVector(js)
    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = 0.0
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= Vector(vec(sum(∇[:, nodes[1].is], dims=2)))
    nodes[1].gain = EvoTrees.get_gain(params, nodes[1].∑) # should use a GPU version?

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
                # @info "length(nodes[n].is)" length(nodes[n].is) depth n
                EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            end
        else
            # update histograms
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    update_hist_gpu!(nodes[n].h, h∇_cpu, h∇, ∇, x_bin, nodes[n].is, jsg, js)
                end
            end
            Threads.@threads for n ∈ n_current
                EvoTrees.update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
            end

            # initialize gains for node 1 in which all gains of a given depth will be accumulated
            if depth > 1
                @inbounds for j in js
                    nodes[1].gains[j] .= 0
                end
            end
            gain = 0
            # update gains based on the aggregation of all nodes of a given depth. One gains matrix per depth (vs one per node in binary trees).
            for n ∈ sort(n_current)
                if n > 1 # accumulate gains in node 1
                    for j in js
                        nodes[1].gains[j] .+= nodes[n].gains[j]
                    end
                end
                gain += nodes[n].gain
            end
            for n ∈ sort(n_current)
                if n > 1
                    for j in js
                        nodes[1].gains[j] .*= nodes[n].gains[j] .> 0 #mask ignore gains if any node isn't eligible (too small per leaf weight)
                    end
                end
            end
            # find best split
            best = findmax(findmax.(nodes[1].gains))
            best_gain = best[1][1]
            best_bin = best[1][2]
            best_feat = best[2]
            if best_gain > gain + params.gamma
                for n in sort(n_current)
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[best_feat][best_bin]
                    tree.split[n] = best_bin != 0

                    _left, _right = split_set_threads_gpu!(
                        out,
                        left,
                        right,
                        nodes[n].is,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )

                    offset += length(nodes[n].is)
                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[best_feat][:, best_bin]
                    nodes[n<<1+1].∑ .= nodes[n].hR[best_feat][:, best_bin]
                    nodes[n<<1].gain = EvoTrees.get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = EvoTrees.get_gain(params, nodes[n<<1+1].∑)

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
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over current nodes for a given depth

    return nothing
end
