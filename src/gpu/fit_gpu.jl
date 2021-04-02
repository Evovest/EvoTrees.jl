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
    𝑛 = CUDA.ones(eltype(𝑖_), length(𝑖_))

    # initialize gradients and weights
    δ = CUDA.ones(T, X_size[1], 2 * K + 1)

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = CuArray(binarize(X, edges))

    # initializde histograms
    hist = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)

    # initialize train nodes
    train_nodes = Vector{TrainNodeGPU{T,UInt32,Vector{T}}}(undef, 2^params.max_depth - 1)

    # store cache
    cache = (params = deepcopy(params),
        X = X, Y = Y, Y_cpu = Y_cpu, K = K,
        pred_gpu = pred_gpu, pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑛 = 𝑛,
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
        train_nodes[1] = TrainNodeGPU(S(0), S(1), ∑, gain)
        tree = grow_tree(cache.δ, cache.hist, params, cache.K, train_nodes, cache.edges, cache.X_bin, 𝑖, 𝑗, 𝑛)
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
    train_nodes::Vector{TrainNodeGPU{T,S,V}},
    edges, X_bin,
    𝑖::I, 𝑗::I, 𝑛::I) where {T,U,R,S,V,I}

    leaf_count = one(S)
    tree_depth = one(S)
    tree = TreeGPU(Vector{TreeNodeGPU{T,S,Bool}}())

    for depth in 1:(params.max_depth-1)
        update_hist_gpu!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛, K, MAX_THREADS=512)        
        # best = find_split_gpu!(hist, edges, node.𝑗, params)
        # nodeid = update_nodeid!(nodeis, 𝑖, best)
    end # end of depth
    return tree
end