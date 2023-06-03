"""
    init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)
    
Initialise EvoTree
"""
function init_evotree_df(
    params::EvoTypes{L,T},
    dtrain::AbstractDataFrame;
    target_name,
    fnames_num=nothing,
    fnames_cat=nothing,
    w_name=nothing,
    offset_name=nothing,
    group_name=nothing
) where {L,T}

    levels = nothing
    offset = !isnothing(offset_name) ? T.(dtrain[:, offset_name]) : nothing
    if L == Logistic
        K = 1
        y = T.(dtrain[!, target_name])
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        K = 1
        y = T.(dtrain[!, target_name])
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == Softmax
        if eltype(dtrain[!, target_name]) <: CategoricalValue
            levels = CategoricalArrays.levels(dtrain[:, target_name])
            y = UInt32.(CategoricalArrays.levelcode.(dtrain[:, target_name]))
        else
            levels = sort(unique(dtrain[!, target_name]))
            yc = CategoricalVector(dtrain[:, target_name], levels=levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
        K = length(levels)
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        K = 2
        y = T.(dtrain[!, target_name])
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        K = 2
        y = T.(dtrain[!, target_name])
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        K = 1
        y = T.(dtrain[!, target_name])
        μ = [mean(y)]
    end
    μ = T.(μ)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    # initialize preds
    nobs = nrow(dtrain)
    pred = zeros(T, K, nobs)
    @threads for i in axes(pred, 2)
        @inbounds view(pred, :, i) .= μ
    end
    !isnothing(offset) && (pred .+= offset')

    # init EvoTree
    bias = [Tree{L,K,T}(μ)]

    _w_name = isnothing(w_name) ? "" : [string(w_name)]
    _offset_name = isnothing(offset_name) ? "" : string(offset_name)

    if isnothing(fnames_cat)
        fnames_cat = String[]
    else
        isa(fnames_cat, String) ? fnames_cat = [fnames_cat] : nothing
        fnames_cat = string.(fnames_cat)
        @assert isa(fnames_cat, Vector{String})
        for name in fnames_cat
            @assert typeof(dtrain[!, name]) <: AbstractCategoricalVector "$name should be <: AbstractCategoricalVector"
            @assert !isordered(dtrain[!, name]) "fnames_cat are expected to be unordered - $name is ordered"
        end
        fnames_cat = string.(fnames_cat)
    end

    if isnothing(fnames_num)
        fnames_num = String[]
        for name in names(dtrain)
            if eltype(dtrain[!, name]) <: Number
                push!(fnames_num, name)
            end
        end
        fnames_num = setdiff(fnames_num, union(fnames_cat, [target_name], [_w_name], [_offset_name]))
    else
        isa(fnames_num, String) ? fnames_num = [fnames_num] : nothing
        fnames_num = string.(fnames_num)
        @assert isa(fnames_num, Vector{String})
        for name in fnames_num
            # @info name
            # @info eltype(dtrain[!, name])
            @assert eltype(dtrain[!, name]) <: Number
        end
    end

    fnames = vcat(fnames_num, fnames_cat)
    nfeats = length(fnames)

    # initialize gradients and weights
    ∇ = zeros(T, 2 * K + 1, nobs)
    w = isnothing(w_name) ? ones(T, size(y)) : Vector{T}(dtrain[!, w_name])
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(dtrain; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = binarize(dtrain; fnames, edges)

    is_in = zeros(UInt32, nobs)
    is_out = zeros(UInt32, nobs)
    mask = zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))

    # initialize histograms
    nodes = [TrainNode(featbins, K, view(is_in, 1:0), T) for n = 1:2^params.max_depth-1]
    out = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :fnames_num => fnames_num,
        :fnames_cat => fnames_cat,
        :fnames => fnames,
        :target_name => target_name,
        :w_name => w_name,
        :offset_name => offset_name,
        :group_name => group_name,
        :levels => levels,
        :edges => edges,
        :fnames => fnames,
        :feattypes => feattypes,
    )

    # initialize model
    m = EvoTree{L,K,T}(bias, info)

    cache = (
        info=Dict(:nrounds => 0),
        x_bin=x_bin,
        y=y,
        w=w,
        K=K,
        nodes=nodes,
        pred=pred,
        is_in=is_in,
        is_out=is_out,
        mask=mask,
        js_=js_,
        js=js,
        out=out,
        left=left,
        right=right,
        ∇=∇,
        edges=edges,
        fnames=fnames,
        featbins=featbins,
        feattypes=feattypes,
        monotone_constraints=monotone_constraints,
    )
    return m, cache
end

"""
    grow_evotree!(evotree::EvoTree{L,K,T}, cache, params::EvoTypes{L,T}) where {L,K,T}

Given a instantiate
"""
function grow_evotree!(evotree::EvoTree{L,K,T}, cache, params::EvoTypes{L,T}) where {L,K,T}

    # compute gradients
    update_grads!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    cache.nodes[1].is = subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    # subsample cols
    sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    # instantiate a tree then grow it
    tree = Tree{L,K,T}(params.max_depth)
    grow_tree!(
        tree,
        cache.nodes,
        params,
        cache.∇,
        cache.edges,
        cache.js,
        cache.out,
        cache.left,
        cache.right,
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints
    )
    push!(evotree.trees, tree)
    predict!(cache.pred, tree, cache.x_bin, cache.feattypes)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single tree
function grow_tree!(
    tree::Tree{L,K,T},
    nodes::Vector{N},
    params::EvoTypes{L,T},
    ∇::Matrix{T},
    edges,
    js,
    out,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,T,N}

    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = T(0)
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # reset
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= @views vec(sum(∇[:, nodes[1].is], dims=2))
    nodes[1].gain = get_gain(params, nodes[1].∑)
    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    # @info "hist"
                    update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
                end
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            else
                # @info "gains & max"
                update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
                best = findmax(findmax.(nodes[n].gains))
                best_gain = best[1][1]
                best_bin = best[1][2]
                best_feat = best[2]
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                    popfirst!(n_next)
                else
                    _left, _right = split_set_threads!(
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
                    nodes[n<<1].gain = get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(params, nodes[n<<1+1].∑)
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


"""
    fit_evotree(params;
        x_train, y_train, w_train=nothing, offset_train=nothing;
        x_eval=nothing, y_eval=nothing, w_eval=nothing, offset_eval=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` input data. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount and EvoTreeMLE.

# Keyword arguments

- `x_train::Matrix`: training data of size `[#observations, #features]`. 
- `y_train::Vector`: vector of train targets of length `#observations`.
- `w_train::Vector`: vector of train weights of length `#observations`. If `nothing`, a vector of ones is assumed.
- `offset_train::VecOrMat`: offset for the training data. Should match the size of the predictions.
- `x_eval::Matrix`: evaluation data of size `[#observations, #features]`. 
- `y_eval::Vector`: vector of evaluation targets of length `#observations`.
- `w_eval::Vector`: vector of evaluation weights of length `#observations`. Defaults to `nothing` (assumes a vector of 1s).
- `offset_eval::VecOrMat`: evaluation data offset. Should match the size of the predictions.
- `metric`: The evaluation metric that wil be tracked on `x_eval`, `y_eval` and optionally `w_eval` / `offset_eval` data. 
    Supported metrics are: 
    - `:mse`: mean-squared error. Adapted for general regression models.
    - `:rmse`: root-mean-squared error (CPU only). Adapted for general regression models.
    - `:mae`: mean absolute error. Adapted for general regression models.
    - `:logloss`: Adapted for `:logistic` regression models.
    - `:mlogloss`: Multi-class cross entropy. Adapted to `EvoTreeClassifier` classification models. 
    - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
    - `:gamma`: Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
    - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
    - `:gaussian_mle`: Gaussian log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :gaussian_mle`. 
    - `:logistic_mle`: Logistic log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :logistic_mle`. 
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `fnames`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
- `return_logger::Bool = false`: if set to true (default), `fit_evotree` return a tuple `(m, logger)` where logger is a dict containing various tracking information.
"""
function fit_evotree_df(
    params::EvoTypes{L,T};
    dtrain::AbstractDataFrame,
    deval=nothing,
    target_name,
    fnames_num=nothing,
    fnames_cat=nothing,
    w_name=nothing,
    offset_name=nothing,
    group_name=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    return_logger=false
) where {L,T}

    verbosity == 1 && @info params

    # initialize model and cache
    if String(params.device) == "gpu"
        m, cache = init_evotree_gpu(params, dtrain; target_name, fnames_num, fnames_cat, w_name, offset_name, group_name)
    else
        m, cache = init_evotree_df(params, dtrain; target_name, fnames_num, fnames_cat, w_name, offset_name, group_name)
    end

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(deval)
    any_flag = !isnothing(metric) || !isnothing(deval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, both `metric` and `deval` must be provided."
    end
    logger = nothing
    if logging_flag
        cb = CallBack(params, m; deval, target_name, w_name, offset_name, metric)
        logger = init_logger(;
            T,
            metric,
            maximise=is_maximise(cb.feval),
            early_stopping_rounds
        )
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    end

    # train loop
    for iter = 1:params.nrounds
        grow_evotree!(m, cache, params)
        if !isnothing(logger)
            cb(logger, iter, m.trees[end])
            if iter % print_every_n == 0 && verbosity > 0
                @info "iter $iter" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end # end of callback
    end
    if String(params.device) == "gpu"
        GC.gc(true)
        CUDA.reclaim()
    end
    if return_logger
        return (m, logger)
    else
        return m
    end
end
