function init_evotree_gpu(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector, W = nothing) where {T,U,S}

    K = 1
    levels = nothing
    X = convert(Matrix{T}, X)

    if typeof(params.loss) == Logistic
        Y = CuArray(T.(Y))
        μ = [logit(mean(Y))]
    elseif typeof(params.loss) == Poisson
        Y = CuArray(T.(Y))
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y = CuArray(T.(Y))
        μ = [mean(Y), log(std(Y))]
    else
        Y = CuArray(T.(Y))
        μ = [mean(Y)]
    end

    # initialize preds
    X_size = size(X)
    pred = CUDA.zeros(T, K, X_size[1])
    pred .= CuArray(μ)

    bias = TreeGPU(CuArray(μ))
    evotree = GBTreeGPU([bias], params, Metric(), K, levels)

    # initialize gradients and weights
    δ𝑤 = CUDA.zeros(T, 2 * K + 1, X_size[1])
    W = isnothing(W) ? CUDA.ones(T, size(Y)) : CuVector{T}(W)
    @assert (length(Y) == length(W) && minimum(W) > 0)
    δ𝑤[end, :] .= W

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = CuArray(binarize(X, edges))

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * X_size[2]))

    # initializde histograms
    nodes = [TrainNodeGPU(X_size[2], params.nbins, K, T) for n = 1:2^params.max_depth-1]
    nodes[1].𝑖 = CUDA.zeros(eltype(𝑖_), ceil(Int, params.rowsample * X_size[1]))
    out = CUDA.zeros(UInt32, length(nodes[1].𝑖))
    left = CUDA.zeros(UInt32, length(nodes[1].𝑖))
    right = CUDA.zeros(UInt32, length(nodes[1].𝑖))

    # store cache
    cache = (params = deepcopy(params),
        X = CuArray(X), X_bin = X_bin, Y = Y, K = K,
        nodes = nodes,
        pred = pred,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑗 = 𝑗, 𝑖 = Array(nodes[1].𝑖),
        out = out, left = left, right = right,
        δ𝑤 = δ𝑤,
        edges = edges)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTreeGPU{T}, cache) where {T}

    # initialize from cache
    params = evotree.params
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i = 1:δnrounds
        # select random rows and cols
        sample!(params.rng, cache.𝑖_, cache.𝑖, replace = false, ordered = true)
        sample!(params.rng, cache.𝑗_, cache.𝑗, replace = false, ordered = true)
        cache.nodes[1].𝑖 .= CuArray(cache.𝑖)

        # build a new tree
        update_grads_gpu!(params.loss, cache.δ𝑤, cache.pred, cache.Y)
        # # assign a root and grow tree
        tree = TreeGPU(params.max_depth, evotree.K, zero(T))
        grow_tree_gpu!(tree, cache.nodes, params, cache.δ𝑤, cache.edges, CuVector(cache.𝑗), cache.out, cache.left, cache.right, cache.X_bin, cache.K)
        push!(evotree.trees, tree)
        # update predctions
        predict!(params.loss, cache.pred, tree, cache.X, cache.K)
    end # end of nrounds
    cache.params.nrounds = params.nrounds
    # return model, cache
    return evotree
end

# grow a single tree - grow through all depth
function grow_tree_gpu!(
    tree::TreeGPU{T},
    nodes,
    params::EvoTypes{T,U,S},
    δ𝑤::AbstractMatrix{T},
    edges,
    𝑗, out, left, right,
    X_bin::AbstractMatrix, K) where {T,U,S}

    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # reset nodes
    @threads for n in eachindex(nodes)
        nodes[n].h .= 0
        nodes[n].∑ .= 0
        nodes[n].gain = -Inf
        fill!(nodes[n].gains, -Inf)
    end

    # initialize summary stats
    nodes[1].∑ .= vec(sum(δ𝑤[:, nodes[1].𝑖], dims = 2))
    nodes[1].gain = get_gain(params.loss, Array(nodes[1].∑), params.lambda, K) # should use a GPU version?

    # grow while there are remaining active nodes - TO DO histogram substraction hits issue on GPU
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        if depth < params.max_depth
            for n_id ∈ 1:length(n_current)
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
                    update_hist_gpu!(params.loss, nodes[n].h, δ𝑤, X_bin, nodes[n].𝑖, 𝑗, K)
                end
            end
        end

        # grow while there are remaining active nodes
        for n ∈ sort(n_current)
            if depth == params.max_depth || @allowscalar(nodes[n].∑[end] <= params.min_weight)
                pred_leaf_gpu!(params.loss, tree.pred, n, Array(nodes[n].∑), params)
            else
                update_gains_gpu!(params.loss, nodes[n], 𝑗, params, K)
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
                    pred_leaf_gpu!(params.loss, tree.pred, n, Array(nodes[n].∑), params)
                    popfirst!(n_next)
                else
                    _left, _right = split_set_threads_gpu!(out, left, right, @allowscalar(nodes[n]).𝑖, X_bin, @allowscalar(tree.feat[n]), @allowscalar(tree.cond_bin[n]), offset)
                    nodes[n<<1].𝑖, nodes[n<<1+1].𝑖 = _left, _right
                    offset += length(nodes[n].𝑖)
                    # println("length(_left): ", length(_left), " | length(_right): ", length(_right))
                    # println("best: ", best)
                    update_childs_∑_gpu!(params.loss, nodes, n, best[2][1], best[2][2])
                    nodes[n<<1].gain = get_gain(params.loss, Array(nodes[n<<1].∑), params.lambda, K)
                    nodes[n<<1+1].gain = get_gain(params.loss, Array(nodes[n<<1+1].∑), params.lambda, K)

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
