"""
    grow_evotree!(evotree::EvoTree{L,K,T}, cache, params::EvoTypes{L,T}) where {L,K,T}

Given a instantiate
"""
function train!(m::EvoTree{L,K,T}, cache, params::EvoTypes{L,T}) where {L,K,T}

    # compute gradients
    update_grads!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    cache.nodes[1].is = subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    # subsample cols
    sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    # instantiate a tree then grow it
    tree = Tree{L,K,T}(params.max_depth)
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
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints
    )
    push!(m.trees, tree)
    predict!(cache.pred, tree, cache.x_bin, cache.feattypes)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single tree
function grow!(
    tree::Tree{L,K,T},
    nodes::Vector{N},
    params::EvoTypes{L,T},
    ∇::Matrix{T},
    edges,
    js,
    out,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,T,N}

    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = T(0)
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # reset
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= @views vec(sum(∇[:, nodes[1].is], dims=2))
    nodes[1].gain = get_gain(params, nodes[1].∑)
    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    # @info "hist"
                    update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
                end
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            else
                # @info "gains & max"
                update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
                best = findmax(findmax.(nodes[n].gains))
                best_gain = best[1][1]
                best_bin = best[1][2]
                best_feat = best[2]
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                    popfirst!(n_next)
                else
                    _left, _right = split_set_threads!(
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
                    nodes[n<<1].gain = get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(params, nodes[n<<1+1].∑)
                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
                    popfirst!(n_next)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth
    return nothing
end
