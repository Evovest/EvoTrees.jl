function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}

    # compute gradients
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
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
        cache.nidx,
        is,
        cache.js,
        cache.h∇,
        cache.h∇L,
        cache.h∇R,
        cache.gains,
        cache.x_bin,
        cache.cond_feats_gpu,
        cache.cond_bins_gpu,
        cache.feattypes_gpu,
        cache.monotone_constraints_gpu,
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
    nidx,
    is,
    js,
    h∇,
    h∇L,
    h∇R,
    gains,
    x_bin::CuMatrix,
    cond_feats_gpu,
    cond_bins_gpu,
    feattypes_gpu::CuVector{Bool},
    monotone_constraints_gpu,
) where {L,K,N}

    backend = KernelAbstractions.get_backend(x_bin)

    js_gpu = KernelAbstractions.adapt(backend, js)
    is_gpu = KernelAbstractions.adapt(backend, is)

    # reset nodes
    nidx .= 1
    gains .= 0
    for n in nodes
        n.∑ .= 0
        n.gain = 0.0
    end
    h∇ .= 0
    h∇L .= 0
    h∇R .= 0

    # initialize
    anodes = [1]
    depth = 1

    # Pre-allocate backend-specific buffers (reuse every depth)
    max_nodes = Int32(2^(params.max_depth - 1))
    best_gain_gpu  = KernelAbstractions.zeros(backend, eltype(gains), max_nodes)
    best_bin_gpu   = KernelAbstractions.zeros(backend, Int32, max_nodes)
    best_feat_gpu  = KernelAbstractions.zeros(backend, Int32, max_nodes)

    active_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes)

    # initialize summary stats
    nodes[1].∑ .= Vector(vec(sum(∇[:, is], dims=2)))
    nodes[1].gain = EvoTrees.get_gain(params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(anodes) > 0 && depth <= params.max_depth
        n_next = Int[]
        dnodes = 2^(depth-1):2^depth-1
        offset = 2^(depth - 1) - 1 # identifies breakpoint for each node set within a depth

        if depth < params.max_depth
            update_hist_gpu_optimized!(h∇, ∇, x_bin, is_gpu, js_gpu, nidx, depth, anodes)
            # Compute gains on GPU
            n_active = length(dnodes)
            active_nodes = view(active_nodes_gpu, 1:n_active)
            # copy dnodes (range) into device slice
            copyto!(active_nodes, Int32.(dnodes))
            kernel! = compute_gains_kernel!(backend)
            kernel!(gains, h∇, active_nodes, eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                    ndrange=(n_active, size(h∇, 3)))
            KernelAbstractions.synchronize(backend)
            # update cumulative histograms for child stats
            cumsum!(h∇L, h∇; dims=2)
            h∇R .= h∇L
            reverse!(h∇R; dims=2)
            h∇R .= view(h∇R, :, 1:1, :, :) .- h∇L

            # Compute best split per node on GPU to avoid large host transfer
            # Use pre-allocated buffers (first length(active_nodes) elements)
            view_gain = view(best_gain_gpu, 1:n_active)
            view_bin  = view(best_bin_gpu,  1:n_active)
            view_feat = view(best_feat_gpu, 1:n_active)

            kernel_split! = best_split_kernel!(backend)
            kernel_split!(view_gain, view_bin, view_feat, gains, active_nodes; ndrange=n_active)
            KernelAbstractions.synchronize(backend)

            best_gains = Vector(view_gain)
            best_bins  = Vector(view_bin)
            best_feats = Vector(view_feat)

            for n in anodes
                best_gain = best_gains[n-offset]
                best_bin  = best_bins[n-offset]
                best_feat = best_feats[n-offset]
                # @info "node: $n | best_gain: $best_gain | best_bin: $best_bin | nodegain: $(nodes[n].gain)"
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                    tree.split[n] = best_bin != 0

                    copyto!(nodes[n<<1].∑, view(h∇L, :, best_bin, best_feat, n))
                    copyto!(nodes[n<<1+1].∑, view(h∇R, :, best_bin, best_feat, n))
                    nodes[n<<1].gain = EvoTrees.get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = EvoTrees.get_gain(params, nodes[n<<1+1].∑)
                    push!(n_next, n << 1)
                    push!(n_next, (n << 1) + 1)
                else
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
                end
            end
            copyto!(view(cond_feats_gpu, dnodes), tree.feat[dnodes])
            copyto!(view(cond_bins_gpu, dnodes), tree.cond_bin[dnodes])
            # @info "cond_bins_gpu[dnodes]" Int(minimum(cond_bins_gpu[dnodes]))
            update_nodes_idx_gpu!(nidx, is_gpu, x_bin, cond_feats_gpu, cond_bins_gpu, feattypes_gpu)
        else
            for n in anodes
                EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
            end
        end
        anodes = copy(n_next)
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

    backend = KernelAbstractions.get_backend(x_bin)
    jsg = KernelAbstractions.adapt(backend, js)
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
                EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
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
                        push!(n_next, (n << 1) + 1)
                    else
                        push!(n_next, (n << 1) + 1)
                        push!(n_next, n << 1)
                    end
                end
            else
                for n in n_current
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over current nodes for a given depth

    return nothing
end

