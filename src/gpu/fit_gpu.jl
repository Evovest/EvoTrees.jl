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
    pred_cpu = zeros(T, K, X_size[1])
    pred_gpu = CUDA.zeros(T, K, X_size[1])
    pred_cpu .= μ'
    pred_gpu .= CuArray(μ)'

    bias = TreeGPU(CuArray(μ))
    evotree = GBTreeGPU([bias], params, Metric(), K, levels)

    
    # initialize gradients and weights
    δ𝑤 = CUDA.ones(T, 2 * K + 1, X_size[1])
    
    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = CuArray(binarize(X, edges))
    
    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * X_size[2]))

    # initializde histograms
    nodes = [TrainNode(X_size[2], params.nbins, K, T) for n in 1:2^params.max_depth - 1]
    nodes[1].𝑖 = zeros(eltype(𝑖_), ceil(Int, params.rowsample * X_size[1]))
    left = CUDA.zeros(UInt32, length(nodes[1].𝑖))
    right = CUDA.zeros(UInt32, length(nodes[1].𝑖))

    # initializde histograms
    # hist = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    # histL = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    # histR = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    # gains = CUDA.fill(T(-Inf), params.nbins, X_size[2], 2^params.max_depth - 1)

    # store cache
    cache = (params = deepcopy(params),
        X = X, Y_gpu = Y, Y_cpu = Y_cpu, K = K,
        pred_gpu = pred_gpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑖 = 𝑖, 𝑗 = 𝑗,
        left = left, right = right,
        δ𝑤 = δ𝑤,
        edges = edges, 
        X_bin = X_bin)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTreeGPU{T,S}, cache; verbosity=1) where {T,S}

    # initialize from cache
    params = evotree.params
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds
        # select random rows and cols
        sample!(params.rng, cache.𝑖_, cache.𝑖, replace=false, ordered=true)
        sample!(params.rng, cache.𝑗_, cache.𝑗, replace=false, ordered=true)
        # build a new tree
        update_grads_gpu!(params.loss, cache.δ𝑤, cache.pred_gpu, cache.Y_gpu)
        # # assign a root and grow tree
        tree = TreeGPU(UInt32(params.max_depth), evotree.K, params.λ)
        grow_tree_gpu!(tree, params, cache.δ, cache.hist, cache.histL, cache.histR, cache.gains, cache.edges, CuVector(cache.𝑖), CuVector(cache.𝑗), cache.X_bin);
        push!(evotree.trees, tree)
        # bad GPU usage - to be improved!
        predict_gpu!(cache.pred_gpu, tree, cache.X_bin)
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
    hist::AbstractArray{T,4}, histL::AbstractArray{T,4}, histR::AbstractArray{T,4},
    gains::AbstractArray{T,3},
    edges,
    𝑗, left, right,
    X_bin::AbstractMatrix) where {T,U,S}

    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= vec(sum(δ𝑤[:, nodes[1].𝑖], dims=2))
    nodes[1].gain = get_gain_gpu(params.loss, nodes[1].∑, params.λ)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
    # for depth in 1:(params.max_depth - 1)
        for n ∈ n_current
            if depth == params.max_depth
                pred_leaf_gpu!(params.loss, tree.pred, n, nodes[n].∑, params)
            else
                # histogram subtraction
                if n > 1 && n % 2 == 1
                    nodes[n].h .= nodes[n >> 1].h .- nodes[n - 1].h
                else
                    update_hist_gpu!(params.loss, nodes[n].h, δ𝑤, X_bin, nodes[n].𝑖, 𝑗)
                end
                update_gains_gpu!(params.loss, nodes[n], 𝑗, params)
                best = findmax(nodes[n].gains)
                if best[2][1] != params.nbins && best[1] > nodes[n].gain + params.γ
                    tree.gain[n] = best[1]
                    tree.cond_bin[n] = best[2][1]
                    tree.feat[n] = best[2][2]
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    pred_leaf_gpu(params.loss, tree.pred, n, nodes[n].∑, params)
                    popfirst!(n_next)
                else
                    _left, _right = split_set_gpu!(left, right, nodes[n].𝑖, X_bin, tree.feat[n], tree.cond_bin[n]) # likely need to set a starting point so that remaining split_set withing depth don't override the view
                    nodes[n << 1].𝑖 = _left
                    nodes[n << 1 + 1].𝑖 = _right
                    update_childs_∑_gpu!(params.loss, nodes, n, best[2][1], best[2][2])
                    nodes[n << 1].gain = get_gain_gpu(params.loss, nodes[n << 1].∑, params.λ)
                    nodes[n << 1 + 1].gain = get_gain_gpu(params.loss, nodes[n << 1 + 1].∑, params.λ)
                    push!(n_next, n << 1)
                    push!(n_next, n << 1 + 1)
                    popfirst!(n_next)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth
    return nothing
end
