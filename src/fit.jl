# initialise evotree
function init_evotree(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; fnames=nothing, verbosity=1) where {T,U,S}

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)

    if typeof(params.loss) == Logistic
        Y = T.(Y)
        μ = [logit(mean(Y))]
    elseif typeof(params.loss) == Poisson
        Y = T.(Y)
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Softmax
        if eltype(Y) <: CategoricalValue
            levels = CategoricalArrays.levels(Y)
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(CategoricalArrays.levelcode.(Y))
        else
            levels = sort(unique(Y))
            yc = CategoricalVector(Y, levels=levels)
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y = T.(Y)
        μ = [mean(Y), log(std(Y))]
    else
        Y = T.(Y)
        μ = [mean(Y)]
    end

    # initialize preds
    X_size = size(X)
    pred_cpu = zeros(T, X_size[1], K)
    @inbounds for i in eachindex(pred_cpu)
        pred_cpu[i,:] .= μ
    end

    bias = Tree([TreeNode(μ)])
    evotree = GBTree([bias], params, Metric(), K, levels)

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑛 = ones(UInt32, length(𝑖_))

    # initialize gradients and weights
    δ = ones(T, X_size[1], 2 * K + 1)

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initializde histograms
    hist = zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    histL = zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    histR = zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    gains = fill(T(-Inf), params.nbins, X_size[2], 2^params.max_depth - 1)
    
    # initialize train nodes
    nodes = (gains = fill(T(-Inf), 2^(params.max_depth - 1) - 1),
        feats = zeros(Int, 2^(params.max_depth - 1) - 1),
        cond_bins = zeros(UInt8, 2^(params.max_depth - 1) - 1),
        cond_floats = zeros(T, 2^(params.max_depth - 1) - 1),
        preds = [zeros(T, K) for i in 1:(2^params.max_depth - 1)])

    cache = (params = deepcopy(params),
        X = X, Y_cpu = Y, K = K,
        pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑛 = 𝑛,
        δ = δ,
        edges = edges, 
        X_bin = X_bin,
        gains = gains,
        nodes = nodes,
        # train_nodes = train_nodes,
        hist = hist, histL = histL, histR = histR)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree{T,S}, cache; verbosity=1) where {L,T,S}

    # initialize from cache
    params = evotree.params
    train_nodes = cache.train_nodes
    splits = cache.splits
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = cache.𝑖_[sample(params.rng, cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = cache.𝑗_[sample(params.rng, cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]
        # reset gain to -Inf

        # build a new tree
        update_grads!(params.loss, params.α, cache.pred_cpu, cache.Y_cpu, cache.δ, cache.δ², cache.𝑤)
        # ∑δ, ∑δ², ∑𝑤 = sum(cache.δ[𝑖]), sum(cache.δ²[𝑖]), sum(cache.𝑤[𝑖])
        # gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)
        # assign a root and grow tree
        # train_nodes[1] = TrainNode(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.hist, cache.histL, cache.histR, params, cache.gains, cache.nodes, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        predict!(cache.pred_cpu, tree, cache.X)

    end # end of nrounds

    cache.params.nrounds = params.nrounds
    return evotree
end

# grow a single tree
function grow_tree(
    δ::Matrix{T},
    hist::AbstractArray{T,4}, histL::AbstractArray{T,4}, histR::AbstractArray{T,4},
    params::EvoTypes{T,U,S},
    gains::AbstractArray{T,3},
    nodes,
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
        update_hist!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)
        update_gains!(gains, hist, histL, histR, 𝑗, params, nid)
        # best = findmax(view(gains, :,:,nid))
        # println("best: ", best)
        @inbounds for n in nid
            best = findmax(view(gains, :,:,n))
            # println("best: ", best)
            nodes[:gains][n] = best[1]
            nodes[:feats][n] = best[2][2]
            nodes[:cond_bins][n] = best[2][1]
            # findmax!(bval, bidx, view(gains, :,:,n))
            # nodes[:gains][n] = bval[1]
            # nodes[:feats][n] = bidx[2]
            # nodes[:cond_bins][n] = bidx[1]
        end
        update_set!(𝑛, 𝑖, X_bin, nodes[:feats], nodes[:cond_bins], params.nbins)
    end # end of loop over active ids for a given depth
    return nothing
end


function fit_evotree(params, X_train, Y_train;
    X_eval=nothing, Y_eval=nothing,
    early_stopping_rounds=9999,
    eval_every_n=1,
    print_every_n=9999,
    verbosity=1)

    # initialize metric
    iter_since_best = 0
    if params.metric != :none
        metric_track = Metric()
        metric_best = Metric()
    end

    nrounds_max = params.nrounds
    params.nrounds = 0
    if params.device == "gpu"
        model, cache = init_evotree_gpu(params, X_train, Y_train)
    else 
        model, cache = init_evotree(params, X_train, Y_train)
    end

    iter = 1
    if params.metric != :none && X_eval !== nothing
        pred_eval = predict(model.trees[1], X_eval, model.K)
        Y_eval = convert.(eltype(cache.Y_cpu), Y_eval)
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree!(model, cache)
        # callback function
        if params.metric != :none
            if X_eval !== nothing
                predict!(pred_eval, model.trees[model.params.nrounds + 1], X_eval)
                metric_track.metric = eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric = eval_metric(Val{params.metric}(), cache.pred_cpu, cache.Y_cpu, params.α)
            end
            if metric_track.metric < metric_best.metric
                metric_best.metric = metric_track.metric
                metric_best.iter =  model.params.nrounds
                iter_since_best = 0
            else
                iter_since_best += 1
            end
            if mod(model.params.nrounds, print_every_n) == 0 && verbosity > 0
                display(string("iter:", model.params.nrounds, ", eval: ", metric_track.metric))
            end
        end # end of callback
    end
    if params.metric != :none
        model.metric.iter = metric_best.iter
        model.metric.metric = metric_best.metric
    end
    params.nrounds = nrounds_max
    return model
end
