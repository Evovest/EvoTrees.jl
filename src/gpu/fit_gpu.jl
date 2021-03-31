# initialise evotree
function init_evotree_gpu(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; verbosity=1) where {T,U,S}

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)
    if typeof(params.loss) == Logistic
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [logit(mean(Y))]
    elseif typeof(params.loss) == Poisson
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Softmax
        if typeof(Y) <: AbstractCategoricalVector
            levels = CategoricalArray(CategoricalArrays.levels(Y))
            K = length(levels)
            μ = zeros(T, K)
            Y = MLJModelInterface.int.(Y)
        else
            levels = CategoricalArray(sort(unique(Y)))
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(Y)
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [mean(Y), log(std(Y))]
    else
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [mean(Y)]
    end

    # initialize preds
    X_size = size(X)
    pred_cpu = zeros(T, X_size[1], K)
    pred_gpu = CUDA.zeros(T, X_size[1], K)
    pred_cpu .= μ'
    pred_gpu .= CuArray(μ)'

    bias = TreeGPU([TreeNodeGPU(μ)])
    evotree = GBTreeGPU([bias], params, Metric(), UInt32(K), levels)

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))

    # initialize gradients and weights
    δ = CUDA.ones(T, X_size[1], 2 * K + 1)

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin_cpu = binarize(X, edges)
    X_bin = CuArray(X_bin_cpu)

    # initializde histograms
    hist = [CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2]) for i in 1:2^params.max_depth - 1]

    # initialize train nodes
    train_nodes = Vector{TrainNodeGPU{T,UInt32,CuVector{UInt32},Vector{T}}}(undef, 2^params.max_depth - 1)

    # store cache
    cache = (params = deepcopy(params),
        X = X, Y = Y, Y_cpu = Y_cpu, K = K,
        pred_gpu = pred_gpu, pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 
        δ = δ,
        edges = edges,
        X_bin = X_bin,
        train_nodes = train_nodes,
        # splits = splits,
        hist = hist)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTreeGPU{T,S}, cache; verbosity=1) where {T,S}

    # initialize from cache
    params = evotree.params
    train_nodes = cache.train_nodes
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = CuVector(cache.𝑖_[sample(params.rng, cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)])
        𝑗 = CuVector(cache.𝑗_[sample(params.rng, cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)])

        # build a new tree
        update_grads_gpu!(params.loss, cache.δ, cache.pred_gpu, cache.Y)

        # ∑ = vec(sum(cache.δ[𝑖,:], dims=1))
        ∑ = Array(vec(sum(cache.δ[𝑖,:], dims=1)))

        gain = get_gain_gpu(params.loss, ∑, params.λ)
        # # assign a root and grow tree
        train_nodes[1] = TrainNodeGPU(S(0), S(1), ∑, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.hist, params, cache.K, train_nodes, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        # bad GPU usage - to be improved!
        predict!(cache.pred_cpu, tree, cache.X)
        cache.pred_gpu .= CuArray(cache.pred_cpu)

    end # end of nrounds

    cache.params.nrounds = params.nrounds
    # return model, cache
    return evotree
end

# grow a single tree - grow through all depth
function grow_tree(δ, hist, 
    params::EvoTypes{T,U,R}, K,
    train_nodes::Vector{TrainNodeGPU{T,S,I,V}},
    edges, X_bin) where {T,U,R,S,I,V}

    active_id = ones(S, 1)
    leaf_count = one(S)
    tree_depth = one(S)
    tree = TreeGPU(Vector{TreeNodeGPU{T,S,Bool}}())

    hist_cpu = zeros(T, size(hist[1]))

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(S, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑[end] <= params.min_weight + 0.1 # rounding needed from histogram substraction
                push!(tree.nodes, TreeNodeGPU(pred_leaf_gpu(params.loss, node, params)))
            else
                
                if id > 1 && id == tree.nodes[node.parent].right
                    # hist[id] = hist[node.parent] - hist[id - 1]
                    update_hist_gpu!(hist[id], δ, X_bin, node.𝑖, node.𝑗, K)
                else
                    update_hist_gpu!(hist[id], δ, X_bin, node.𝑖, node.𝑗, K)
                end

                best = find_split_gpu!(hist[id], edges, node.𝑗, params)
                # grow node if best split improves gain
                if best[:gain] > node.gain + params.γ + 1e-5 && best[:∑L][end] > 1e-5 && best[:∑R][end] > 1e-5
                    # if best[:gain] > node.gain + params.γ
    
                    left, right = update_set_gpu(node.𝑖, best[:bin], X_bin[:, best[:feat]])
                    train_nodes[leaf_count + 1] = TrainNodeGPU(id, node.depth + S(1), best[:∑L], best[:gainL], left, node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNodeGPU(id, node.depth + S(1), best[:∑R], best[:gainR], right, node.𝑗)
                    push!(tree.nodes, TreeNodeGPU(leaf_count + S(1), leaf_count + S(2), best[:feat], best[:cond], best[:gain] - node.gain, K))
    
                    push!(next_active_id, leaf_count + S(1))
                    push!(next_active_id, leaf_count + S(2))
                    leaf_count += S(2)
                else
                    push!(tree.nodes, TreeNodeGPU(pred_leaf_gpu(params.loss, node, params)))
                end # end of single node split search
            end
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += S(1)
    end # end of tree growth
    return tree
end