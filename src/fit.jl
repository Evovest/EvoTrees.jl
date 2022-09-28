"""
    init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)
    
Initialise EvoTree
"""
function init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W=nothing, offset=nothing) where {T,U,S}

    K = 1
    levels = nothing
    X = convert(Matrix{T}, X)

    if typeof(params.loss) == Logistic
        Y = T.(Y)
        μ = [logit(mean(Y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif typeof(params.loss) ∈ [Poisson, Gamma, Tweedie]
        Y = T.(Y)
        μ = fill(log(mean(Y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
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
        !isnothing(offset) && offset .= log.(offset)
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y = T.(Y)
        μ = [mean(Y), log(std(Y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        Y = T.(Y)
        μ = [mean(Y)]
    end

    # force a neutral bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    # initialize preds
    X_size = size(X)
    pred = zeros(T, K, X_size[1])
    @inbounds for i = 1:X_size[1]
        pred[:, i] .= μ
    end
    !isnothing(offset) && (pred .+= offset')

    bias = Tree(μ)
    evotree = GBTree([bias], params, Metric(), K, levels)

    # initialize gradients and weights
    δ𝑤 = zeros(T, 2 * K + 1, X_size[1])
    W = isnothing(W) ? ones(T, size(Y)) : Vector{T}(W)
    @assert (length(Y) == length(W) && minimum(W) > 0)
    δ𝑤[end, :] .= W

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))
    𝑗 = zeros(eltype(𝑗_), ceil(Int, params.colsample * X_size[2]))

    # initialize histograms
    nodes = [TrainNode(X_size[2], params.nbins, K, T) for n = 1:2^params.max_depth-1]
    nodes[1].𝑖 = zeros(eltype(𝑖_), ceil(Int, params.rowsample * X_size[1]))
    out = zeros(UInt32, length(nodes[1].𝑖))
    left = zeros(UInt32, length(nodes[1].𝑖))
    right = zeros(UInt32, length(nodes[1].𝑖))

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, X_size[2])
    hasproperty(params, :monotone_constraint) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    cache = (params=deepcopy(params),
        X=X, Y=Y, K=K,
        nodes=nodes,
        pred=pred,
        𝑖_=𝑖_, 𝑗_=𝑗_, 𝑗=𝑗,
        out=out, left=left, right=right,
        δ𝑤=δ𝑤,
        edges=edges,
        X_bin=X_bin,
        monotone_constraints=monotone_constraints)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree{T}, cache) where {T,S}

    # initialize from cache
    params = evotree.params
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i = 1:δnrounds
        # select random rows and cols
        sample!(params.rng, cache.𝑖_, cache.nodes[1].𝑖, replace=false, ordered=true)
        sample!(params.rng, cache.𝑗_, cache.𝑗, replace=false, ordered=true)

        # build a new tree
        update_grads!(params.loss, cache.δ𝑤, cache.pred, cache.Y, params.alpha)
        # assign a root and grow tree
        tree = Tree(params.max_depth, evotree.K, zero(T))
        grow_tree!(tree, cache.nodes, params, cache.δ𝑤, cache.edges, cache.𝑗, cache.out, cache.left, cache.right, cache.X_bin, cache.K, cache.monotone_constraints)
        push!(evotree.trees, tree)
        predict!(params.loss, cache.pred, tree, cache.X, cache.K)

    end # end of nrounds
    cache.params.nrounds = params.nrounds
    return nothing
end

# grow a single tree
function grow_tree!(
    tree::Tree{T},
    nodes::Vector{TrainNode{T}},
    params::EvoTypes{T,U,S},
    δ𝑤::Matrix{T},
    edges,
    𝑗, out, left, right,
    X_bin::AbstractMatrix, K, monotone_constraints) where {T,U,S}

    # reset nodes
    @threads for n in eachindex(nodes)
        [nodes[n].h[j] .= 0 for j in 𝑗]
        nodes[n].∑ .= 0
        nodes[n].gain = 0
        fill!(nodes[n].gains, -Inf)
    end

    # reset
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= vec(sum(δ𝑤[:, nodes[1].𝑖], dims=2))
    nodes[1].gain = get_gain(params.loss, nodes[1].∑, params.lambda, K)
    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth

        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        nodes[n].h .= nodes[n>>1].h .- nodes[n+1].h
                    else
                        nodes[n].h .= nodes[n>>1].h .- nodes[n-1].h
                    end
                else
                    update_hist!(params.loss, nodes[n].h, δ𝑤, X_bin, nodes[n].𝑖, 𝑗, K)
                end
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                pred_leaf_cpu!(params.loss, tree.pred, n, nodes[n].∑, params, K, δ𝑤, nodes[n].𝑖)
            else
                # histogram subtraction
                update_gains!(params.loss, nodes[n], 𝑗, params, K, monotone_constraints)
                best = findmax(nodes[n].gains)
                if best[2][1] != params.nbins && best[1] > nodes[n].gain + params.gamma
                    tree.gain[n] = best[1] - nodes[n].gain
                    tree.cond_bin[n] = best[2][1]
                    tree.feat[n] = best[2][2]
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    pred_leaf_cpu!(params.loss, tree.pred, n, nodes[n].∑, params, K, δ𝑤, nodes[n].𝑖)
                    popfirst!(n_next)
                else
                    # println("typeof(nodes[n].𝑖): ", typeof(nodes[n].𝑖))
                    _left, _right = split_set_threads!(out, left, right, nodes[n].𝑖, X_bin, tree.feat[n], tree.cond_bin[n], offset)
                    nodes[n<<1].𝑖, nodes[n<<1+1].𝑖 = _left, _right
                    offset += length(nodes[n].𝑖)
                    update_childs_∑!(params.loss, nodes, n, best[2][1], best[2][2], K)
                    nodes[n<<1].gain = get_gain(params.loss, nodes[n<<1].∑, params.lambda, K)
                    nodes[n<<1+1].gain = get_gain(params.loss, nodes[n<<1+1].∑, params.lambda, K)

                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
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


"""
    fit_evotree(params, x_train, y_train, w_train=nothing, offset_train=nothing;
        x_eval=nothing, y_eval=nothing, w_eval = nothing, offset_eval=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` input data. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount or EvoTreeGaussian
- `x_train::Matrix`: training data of size `[#observations, #features]`. 
- `y_train::Vector`: vector of train targets of length `#observations`.
- `w_train::Vector`: vector of train weights of length `#observations`. Defaults to `nothing` and a vector of ones is assumed.
- `offset_train::VecOrMat`: offset for the training data. Defaults to `nothing`. Should match the size of the predictions.

# Keyword arguments

- `x_eval::Matrix`: evaluation data of size `[#observations, #features]`. 
- `y_eval::Vector`: vector of evaluation targets of length `#observations`.
- `w_eval::Vector`: vector of evaluation weights of length `#observations`. Defaults to `nothing` (assumes a vector of 1s).
- `offset_eval::VecOrMat`: evaluation data offset. Should match the size of the predictions.
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
"""
function fit_evotree(params::EvoTypes;
    x_train::AbstractMatrix, y_train::AbstractVector, w_train=nothing, offset_train=nothing,
    x_eval=nothing, y_eval=nothing, w_eval=nothing, offset_eval=nothing,
    early_stopping_rounds=9999,
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
        model, cache = init_evotree_gpu(params, x_train, y_train, w_train, offset_train)
        if params.metric != :none && !isnothing(x_eval)
            x_eval = CuArray(eltype(cache.X).(x_eval))
            y_eval = CuArray(eltype(cache.Y).(y_eval))
            w_eval = isnothing(w_eval) ? CUDA.ones(eltype(cache.X), size(y_eval)) : CuArray(eltype(cache.X).(w_eval))
            pred_eval = predict(params.loss, model.trees[1], x_eval, model.K)
            !isnothing(offset_eval) && pred_eval .+= offset_eval
            eval_vec = CUDA.zeros(eltype(cache.pred), size(y_eval, 1))
        elseif params.metric != :none
            eval_vec = CUDA.zeros(eltype(cache.pred), size(y_train, 1))
        end
    else
        model, cache = init_evotree(params, x_train, y_train, w_train, offset_train)
        if params.metric != :none && !isnothing(x_eval)
            pred_eval = predict(params.loss, model.trees[1], x_eval, model.K)
            if !isnothing(offset_eval)
                typeof(params.loss) == Logistic && offset_eval .= logit.(offset_eval)
                typeof(params.loss) in [Poisson, Gamma, Tweedie] && offset_eval .= log.(offset_eval)
                typeof(params.loss) == Softmax && offset_eval .= log.(offset_eval)
                typeof(params.loss) == Softmax && offset_eval[2, :] .= offset_eval[2, :]
                pred_eval .+= offset_eval
            end
            y_eval = convert.(eltype(cache.Y), y_eval)
            w_eval = isnothing(w_eval) ? ones(eltype(cache.X), size(y_eval)) : eltype(cache.X).(w_eval)
        end
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree!(model, cache)
        # callback function
        if params.metric != :none
            if x_eval !== nothing
                predict!(params.loss, pred_eval, model.trees[model.params.nrounds+1], x_eval, model.K)
                if params.device == "gpu"
                    metric_track.metric = eval_metric(Val{params.metric}(), eval_vec, pred_eval, y_eval, w_eval, params.alpha)
                else
                    metric_track.metric = eval_metric(Val{params.metric}(), pred_eval, y_eval, w_eval, params.alpha)
                end
            else
                if params.device == "gpu"
                    metric_track.metric = eval_metric(Val{params.metric}(), eval_vec, cache.pred, cache.Y, cache.δ𝑤[end, :], params.alpha)
                else
                    metric_track.metric = eval_metric(Val{params.metric}(), cache.pred, cache.Y, cache.δ𝑤[end, :], params.alpha)
                end
            end
            if metric_track.metric < metric_best.metric
                metric_best.metric = metric_track.metric
                metric_best.iter = model.params.nrounds
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
