# initialise evotree
function init_evotree(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; fnames=nothing, verbosity=1) where {T,U,S}

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)

    if typeof(params.loss) == Poisson
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
    pred_cpu = zeros(T, K, X_size[1])
    @inbounds for i in eachindex(pred_cpu)
        pred_cpu[:,i] .= μ
    end

    bias = Tree(μ)
    evotree = GBTree([bias], params, Metric(), K, levels)

    # initialize gradients and weights
    δ𝑤 = ones(T, 2 * K + 1, X_size[1])
    
    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)
    
    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * X_size[2]))

    # initializde histograms
    nodes = [TrainNode(X_size[2], params.nbins, K, T) for n in 1:2^params.max_depth - 1]
    nodes[1].𝑖 = zeros(eltype(𝑖_), ceil(Int, params.rowsample * X_size[1]))
    left = zeros(UInt32, length(nodes[1].𝑖))
    right = zeros(UInt32, length(nodes[1].𝑖))

    cache = (params = deepcopy(params),
        X = X, Y_cpu = Y, K = K,
        nodes = nodes,
        pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, 𝑗 = 𝑗,
        left = left, right = right,
        δ𝑤 = δ𝑤,
        edges = edges, 
        X_bin = X_bin)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree{T}, cache; verbosity=1) where {T,S}

    # initialize from cache
    params = evotree.params
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        sample!(params.rng, cache.𝑖_, cache.nodes[1].𝑖, replace=false, ordered=true)
        sample!(params.rng, cache.𝑗_, cache.𝑗, replace=false, ordered=true)

        # build a new tree
        update_grads!(params.loss, cache.δ𝑤, cache.pred_cpu, cache.Y_cpu)
        # ∑δ, ∑δ², ∑𝑤 = sum(cache.δ[𝑖]), sum(cache.δ²[𝑖]), sum(cache.𝑤[𝑖])
        # gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)
        # assign a root and grow tree
        tree = Tree(params.max_depth, evotree.K, zero(T))
        grow_tree!(tree, cache.nodes, params, cache.δ𝑤, cache.edges, cache.𝑗, cache.left, cache.right, cache.X_bin)
        push!(evotree.trees, tree)
        predict!(cache.pred_cpu, tree, cache.X)

    end # end of nrounds
    cache.params.nrounds = params.nrounds
    return evotree
end

# grow a single tree
function grow_tree!(
    tree::Tree{T},
    nodes::Vector{TrainNode{T}},
    params::EvoTypes{T,U,S},
    δ𝑤::Matrix{T},
    edges,
    𝑗, left, right,
    X_bin::AbstractMatrix) where {T,U,S}

    # reset nodes
    for n in eachindex(nodes)
        [nodes[n].h[j] .= 0 for j in eachindex(nodes[n].h)]
        [nodes[n].hL[j] .= 0 for j in eachindex(nodes[n].hL)]
        [nodes[n].hR[j] .= 0 for j in eachindex(nodes[n].hR)]
        nodes[n].∑ .= 0
        nodes[n].gain = 0
        fill!(nodes[n].gains, -Inf)
    end

    # reset
    # bval, bidx = [zero(T)], [(0,0)]
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= vec(sum(δ𝑤[:, nodes[1].𝑖], dims=2))
    nodes[1].gain = get_gain(params.loss, nodes[1].∑, params.λ)
    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
    # for depth in 1:(params.max_depth - 1)
        for n ∈ n_current

            # println("n: ", n)
            if depth == params.max_depth
                # tree.pred[1, n] = pred_leaf_cpu(params.loss, nodes[n].∑, params)
                pred_leaf_cpu!(params.loss, tree.pred, n, nodes[n].∑, params)
            else
                # println("n_current: ", n, " | ", n_current)
                # println("depth: ", depth)
                # histogram subtraction
                if n > 1 && n % 2 == 1
                    nodes[n].h .= nodes[n >> 1].h .- nodes[n - 1].h
                else
                    update_hist!(params.loss, nodes[n].h, δ𝑤, X_bin, nodes[n].𝑖, 𝑗)
                end
                update_gains!(params.loss, nodes[n], 𝑗, params)
                best = findmax(nodes[n].gains)
                # println("best: ", best)
                # println("nodes[n].gain: ", nodes[n].gain)
                if best[2][1] != params.nbins && best[1] > nodes[n].gain + params.γ
                    tree.gain[n] = best[1]
                    tree.cond_bin[n] = best[2][1]
                    tree.feat[n] = best[2][2]
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    # tree.pred[1, n] = pred_leaf_cpu(params.loss, nodes[n].∑, params)
                    pred_leaf_cpu!(params.loss, tree.pred, n, nodes[n].∑, params)
                    popfirst!(n_next)
                    # println("n_next leaf post: ", n, " | ", n_next)
                else
                    _left, _right = split_set!(left, right, nodes[n].𝑖, X_bin, tree.feat[n], tree.cond_bin[n]) # likely need to set a starting point so that remaining split_set withing depth don't override the view
                    nodes[n << 1].𝑖 = _left
                    nodes[n << 1 + 1].𝑖 = _right

                    # println("length(_left): ", length(_left))
                    # println("length(_right): ", length(_right))
                    # set ∑ stats for child nodes
                    nodes[n << 1].∑ .= nodes[n].hL[best[2][2]][(3 * best[2][1] - 2):(3 * best[2][1])]
                    nodes[n << 1 + 1].∑ .= nodes[n].hR[best[2][2]][(3 * best[2][1] - 2):(3 * best[2][1])]

                    nodes[n << 1].gain = get_gain(params.loss, nodes[n << 1].∑, params.λ)
                    nodes[n << 1 + 1].gain = get_gain(params.loss, nodes[n << 1 + 1].∑, params.λ)

                    push!(n_next, n << 1)
                    push!(n_next, n << 1 + 1)
                    popfirst!(n_next)
                    # println("n_next split post: ", n, " | ", n_next)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
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
