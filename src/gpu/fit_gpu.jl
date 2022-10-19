function init_evotree_gpu(
    params::EvoTypes{L,T,S};
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train = nothing,
    offset_train = nothing,
    fnames = nothing,
) where {L,T,S}

    K = 1
    levels = nothing
    x = convert(Matrix{T}, x_train)

    offset = !isnothing(offset_train) ? T.(offset_train) : nothing
    if L == Logistic
        y = CuArray(T.(y_train))
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L ∈ [Poisson, Gamma, Tweedie]
        y = CuArray(T.(y_train))
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianDist
        K = 2
        y = CuArray(T.(y_train))
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        y = CuArray(T.(y_train))
        μ = [mean(y)]
    end

    # force a neutral bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    # initialize preds
    x_size = size(x)
    pred = CUDA.zeros(T, K, x_size[1])
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # init GBTree
    bias = [TreeGPU{L,T}(CuArray(μ))]
    fnames = isnothing(fnames) ? ["feat_$i" for i in axes(x, 2)] : string.(fnames)
    @assert length(fnames) == size(x, 2)
    info = Dict(:fnames => fnames, :levels => levels)
    evotree = GBTreeGPU{L,T,S}(bias, params, Metric(), K, info)

    # initialize gradients and weights
    δ𝑤 = CUDA.zeros(T, 2 * K + 1, x_size[1])
    w = isnothing(w_train) ? CUDA.ones(T, size(y)) : CuVector{T}(w_train)
    @assert (length(y) == length(w) && minimum(w) > 0)
    δ𝑤[end, :] .= w

    # binarize data into quantiles
    edges = get_edges(x, params.nbins)
    x_bin = CuArray(binarize(x, edges))

    𝑖_ = UInt32.(collect(1:x_size[1]))
    𝑗_ = UInt32.(collect(1:x_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * x_size[2]))

    # initializde histograms
    nodes = [TrainNodeGPU(x_size[2], params.nbins, K, T) for n = 1:2^params.max_depth-1]
    nodes[1].𝑖 = CUDA.zeros(eltype(𝑖_), ceil(Int, params.rowsample * x_size[1]))
    out = CUDA.zeros(UInt32, length(nodes[1].𝑖))
    left = CUDA.zeros(UInt32, length(nodes[1].𝑖))
    right = CUDA.zeros(UInt32, length(nodes[1].𝑖))

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, x_size[2])
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    # store cache
    cache = (
        params = deepcopy(params),
        x = CuArray(x),
        x_bin = x_bin,
        y = y,
        K = K,
        nodes = nodes,
        pred = pred,
        𝑖_ = 𝑖_,
        𝑗_ = 𝑗_,
        𝑗 = 𝑗,
        𝑖 = Array(nodes[1].𝑖),
        out = out,
        left = left,
        right = right,
        δ𝑤 = δ𝑤,
        edges = edges,
        monotone_constraints = CuArray(monotone_constraints),
    )

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTreeGPU{L,T,S}, cache) where {L,T,S}

    # initialize from cache
    params = evotree.params
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i = 1:δnrounds
        # select random rows and cols
        sample!(params.rng, cache.𝑖_, cache.𝑖, replace = false, ordered = true)
        sample!(params.rng, cache.𝑗_, cache.𝑗, replace = false, ordered = true)
        cache.nodes[1].𝑖 .= CuArray(cache.𝑖)

        # build a new tree
        update_grads_gpu!(L, cache.δ𝑤, cache.pred, cache.y)
        # # assign a root and grow tree
        tree = TreeGPU{L,T}(params.max_depth, evotree.K, zero(T))
        grow_tree_gpu!(
            tree,
            cache.nodes,
            params,
            cache.δ𝑤,
            cache.edges,
            CuVector(cache.𝑗),
            cache.out,
            cache.left,
            cache.right,
            cache.x_bin,
            cache.K,
            cache.monotone_constraints,
        )
        push!(evotree.trees, tree)
        # update predctions
        predict!(cache.pred, tree, cache.x, cache.K)
    end # end of nrounds
    cache.params.nrounds = params.nrounds
    CUDA.reclaim()
    return evotree
end

# grow a single tree - grow through all depth
function grow_tree_gpu!(
    tree::TreeGPU{L,T},
    nodes,
    params::EvoTypes{L,T,S},
    δ𝑤::AbstractMatrix,
    edges,
    𝑗,
    out,
    left,
    right,
    x_bin::AbstractMatrix,
    K,
    monotone_constraints,
) where {L,T,S}

    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # reset nodes
    for n in eachindex(nodes)
        nodes[n].h .= 0
        nodes[n].∑ .= 0
        nodes[n].gain = -Inf
        fill!(nodes[n].gains, -Inf)
    end

    # initialize summary stats
    nodes[1].∑ .= vec(sum(δ𝑤[:, nodes[1].𝑖], dims = 2))
    nodes[1].gain = get_gain(L, Array(nodes[1].∑), params.lambda, K) # should use a GPU version?

    # grow while there are remaining active nodes - TO DO histogram substraction hits issue on GPU
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        nodes[n].h .= nodes[n>>1].h .- nodes[n+1].h
                        CUDA.synchronize()
                    else
                        nodes[n].h .= nodes[n>>1].h .- nodes[n-1].h
                        CUDA.synchronize()
                    end
                else
                    update_hist_gpu!(L, nodes[n].h, δ𝑤, x_bin, nodes[n].𝑖, 𝑗, K)
                end
            end
        end

        # grow while there are remaining active nodes
        for n ∈ sort(n_current)
            if depth == params.max_depth ||
               @allowscalar(nodes[n].∑[end] <= params.min_weight)
                pred_leaf_gpu!(tree.pred, n, Array(nodes[n].∑), params)
            else
                update_gains_gpu!(nodes[n], 𝑗, params, K, monotone_constraints)
                best = findmax(nodes[n].gains)
                if best[2][1] != params.nbins && best[1] > nodes[n].gain + params.gamma
                    allowscalar() do
                        tree.gain[n] = best[1]
                        tree.cond_bin[n] = best[2][1]
                        tree.feat[n] = best[2][2]
                        tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                    end
                end
                # println("node: ", n, " | best: ", best, " | nodes[n].gain: ", nodes[n].gain)
                @allowscalar(tree.split[n] = tree.cond_bin[n] != 0)
                if !@allowscalar(tree.split[n])
                    pred_leaf_gpu!(tree.pred, n, Array(nodes[n].∑), params)
                    popfirst!(n_next)
                else
                    _left, _right = split_set_threads_gpu!(
                        out,
                        left,
                        right,
                        @allowscalar(nodes[n]).𝑖,
                        x_bin,
                        @allowscalar(tree.feat[n]),
                        @allowscalar(tree.cond_bin[n]),
                        offset,
                    )
                    nodes[n<<1].𝑖, nodes[n<<1+1].𝑖 = _left, _right
                    offset += length(nodes[n].𝑖)
                    # println("length(_left): ", length(_left), " | length(_right): ", length(_right))
                    # println("best: ", best)
                    update_childs_∑_gpu!(L, nodes, n, best[2][1], best[2][2])
                    nodes[n<<1].gain = get_gain(L, Array(nodes[n<<1].∑), params.lambda, K)
                    nodes[n<<1+1].gain =
                        get_gain(L, Array(nodes[n<<1+1].∑), params.lambda, K)

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
