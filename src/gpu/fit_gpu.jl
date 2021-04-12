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

    bias = TreeGPU(CuArray(μ))
    evotree = GBTreeGPU([bias], params, Metric(), UInt32(K), levels)

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑖 = zeros(eltype(𝑖_), ceil(Int, params.rowsample * X_size[1]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * X_size[2]))
    𝑛 = CUDA.ones(eltype(𝑖_), length(𝑖_))

    # initialize gradients and weights
    δ = CUDA.ones(T, X_size[1], 2 * K + 1)

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = CuArray(binarize(X, edges))

    # initializde histograms
    hist = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    histL = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    histR = CUDA.zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    gains = CUDA.fill(T(-Inf), params.nbins, X_size[2], 2^params.max_depth - 1)

    # store cache
    cache = (params = deepcopy(params),
        X = X, Y = Y, Y_cpu = Y_cpu, K = K,
        pred_gpu = pred_gpu, pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑖 = 𝑖, 𝑗 = 𝑗, 𝑛 = 𝑛,
        δ = δ,
        edges = edges, 
        X_bin = X_bin,
        gains = gains,
        hist = hist, histL = histL, histR = histR)

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
function grow_tree_gpu!(
    tree::TreeGPU{T},
    params::EvoTypes{T,U,S},
    δ::AbstractMatrix{T},
    hist::AbstractArray{T,4}, histL::AbstractArray{T,4}, histR::AbstractArray{T,4},
    gains::AbstractArray{T,3},
    edges,
    𝑖, 𝑗, 𝑛,
    X_bin::AbstractMatrix) where {T,U,S}

    # reset
    # bval, bidx = [zero(T)], [(0,0)]
    hist .= 0
    histL .= 0
    histR .= 0
    gains .= -Inf
    𝑛 .= 1

    # grow while there are remaining active nodes
    for depth in 1:(params.max_depth - 1)
        nid = 2^(depth - 1):2^(depth) - 1
        update_hist_gpu!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛, depth, MAX_THREADS=512)
        update_gains_gpu!(gains, hist, histL, histR, 𝑗, params, nid, depth)
        @inbounds for n in nid
            best = findmax(view(gains, :, :, n))
            # println("best: ", best)
            if best[2][1] != params.nbins && best[1] > -Inf
                tree.gain[n] = best[1]
                tree.feat[n] = best[2][2]
                tree.cond_bin[n] = best[2][1]
                tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
            end
            tree.split[n] = tree.cond_bin[n] != 0
            if !tree.split[n]
                tree.pred[1, n] = pred_leaf_gpu(params, histL, 𝑗[1], n)
            end
        end
        update_set_gpu!(𝑛, 𝑖, X_bin, tree.feat, tree.cond_bin, params.nbins)
    end # end of loop over active ids for a given depth

    # loop on final depth to assign preds
    for n in 2^(params.max_depth-1):2^params.max_depth-1
        
    end

    return nothing
end
