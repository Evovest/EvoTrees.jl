"""
    grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTypes{L}, ::Type{<:Device}=CPU) where {L,K}

Given a instantiate
"""
function grow_evotree!(m::EvoTree{L,K}, cache, params::EvoTypes{L}, ::Type{<:Device}=CPU) where {L,K}

    # compute gradients
    update_grads!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    cache.nodes[1].is = subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    # subsample cols
    sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    # instantiate a tree then grow it
    tree = Tree{L,K}(params.max_depth)
    grow! = params.tree_type == "oblivious" ? grow_otree! : grow_tree!
    grow!(
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
    push!(m.trees, tree)
    predict!(cache.pred, tree, cache.x_bin, cache.feattypes)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single tree
function grow_tree!(
    tree::Tree{L,K},
    nodes::Vector{N},
    params::EvoTypes{L},
    ∇::Matrix,
    edges,
    js,
    out,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,N}

    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = 0.0
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= dropdims(sum(Float64, view(∇, :, nodes[1].is), dims=2), dims=2)
    nodes[1].gain = get_gain(params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
                end
            end
            @threads for n ∈ sort(n_current)
                update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
            end
        end

        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            else
                best = findmax(findmax.(nodes[n].gains))
                best_gain = best[1][1]
                best_bin = best[1][2]
                best_feat = best[2]
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                    tree.split[n] = best_bin != 0

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
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth
    return nothing
end


# grow a single oblivious tree
function grow_otree!(
    tree::Tree{L,K},
    nodes::Vector{N},
    params::EvoTypes{L},
    ∇::Matrix,
    edges,
    js,
    out,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,N}

    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = 0.0
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= dropdims(sum(Float64, view(∇, :, nodes[1].is), dims=2), dims=2)
    nodes[1].gain = get_gain(params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        min_weight_flag = false
        for n in n_current
            nodes[n].∑[end] <= params.min_weight ? min_weight_flag = true : nothing
        end
        if depth == params.max_depth || min_weight_flag
            for n in n_current
                # @info "length(nodes[n].is)" length(nodes[n].is) depth n
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            end
        else
            # update histograms
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in js
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
                end
            end
            @threads for n ∈ n_current
                update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
            end

            # initialize gains for node 1 in which all gains of a given depth will be accumulated
            if depth > 1
                @inbounds for j in js
                    nodes[1].gains[j] .= 0
                end
            end
            gain = 0
            # update gains based on the aggregation of all nodes of a given depth. One gains matrix per depth (vs one per node in binary trees).
            for n ∈ sort(n_current)
                if n > 1 # accumulate gains in node 1
                    for j in js
                        nodes[1].gains[j] .+= nodes[n].gains[j]
                    end
                end
                gain += nodes[n].gain
            end
            for n ∈ sort(n_current)
                if n > 1
                    for j in js
                        nodes[1].gains[j] .*= nodes[n].gains[j] .> 0 #mask ignore gains if any node isn't eligible (too small per leaf weight)
                    end
                end
            end
            # find best split
            best = findmax(findmax.(nodes[1].gains))
            best_gain = best[1][1]
            best_bin = best[1][2]
            best_feat = best[2]
            if best_gain > gain + params.gamma
                for n in sort(n_current)
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[best_feat][best_bin]
                    tree.split[n] = best_bin != 0

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
                end
            else
                for n in n_current
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over current nodes for a given depth
    return nothing
end


"""
    fit_evotree(
        params::EvoTypes{L}, 
        dtrain;
        target_name,
        fnames=nothing,
        w_name=nothing,
        offset_name=nothing,
        deval=nothing,
        metric=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1,
        return_logger=false,
        device="cpu")

Main training function. Performs model fitting given configuration `params`, `dtrain`, `target_name` and other optional kwargs. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` can be one of: 
    - [`EvoTreeRegressor`](@ref)
    - [`EvoTreeClassifier`](@ref)
    - [`EvoTreeCount`](@ref)
    - [`EvoTreeMLE`](@ref)
- `dtrain`: A Tables compatible training data (named tuples, DataFrame...) containing features and target variables. 

# Keyword arguments

- `target_name`: name of target variable. 
- `fnames = nothing`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
- `w_name = nothing`: name of the variable containing weights. If `nothing`, common weights on one will be used.
- `offset_name = nothing`: name of the offset variable.
- `deval`: A Tables compatible evaluation data containing features and target variables. 
- `metric`: The evaluation metric that wil be tracked on `deval`. 
    Supported metrics are: 
    - `:mse`: mean-squared error. Adapted for general regression models.
    - `:rmse`: root-mean-squared error (CPU only). Adapted for general regression models.
    - `:mae`: mean absolute error. Adapted for general regression models.
    - `:logloss`: Adapted for `:logistic` regression models.
    - `:mlogloss`: Multi-class cross entropy. Adapted to `EvoTreeClassifier` classification models. 
    - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
    - `:gamma`: Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
    - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
    - `:gaussian_mle`: Gaussian maximum log-likelihood. Adapted to `EvoTreeMLE` models with `loss = :gaussian_mle`. 
    - `:logistic_mle`: Logistic maximum log-likelihood. Adapted to `EvoTreeMLE` models with `loss = :logistic_mle`. 
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `return_logger::Bool = false`: if set to true (default), `fit_evotree` return a tuple `(m, logger)` where logger is a dict containing various tracking information.
- `device="cpu"`: Hardware device to use for computations. Can be either `"cpu"` or `"gpu"`. Following losses are not GPU supported at the moment`
    :l1`, `:quantile`, `:logistic_mle`.
"""
function fit_evotree(
    params::EvoTypes{L},
    dtrain;
    target_name,
    fnames=nothing,
    w_name=nothing,
    offset_name=nothing,
    deval=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    return_logger=false,
    device="cpu"
) where {L}

    @assert Tables.istable(dtrain) "fit_evotree(params, dtrain) only accepts Tables compatible input for `dtrain` (ex: named tuples, DataFrames...)"
    verbosity == 1 && @info params
    @assert string(device) ∈ ["cpu", "gpu"]
    _device = string(device) == "cpu" ? CPU : GPU

    m, cache = init(params, dtrain, _device; target_name, fnames, w_name, offset_name)

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(deval)
    any_flag = !isnothing(metric) || !isnothing(deval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, both `metric` and `deval` must be provided."
    end
    if logging_flag
        cb = CallBack(params, m, deval, _device; target_name, w_name, offset_name, metric)
        logger = init_logger(; metric, maximise=is_maximise(cb.feval), early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    for i = 1:params.nrounds
        grow_evotree!(m, cache, params, _device)
        if !isnothing(logger)
            cb(logger, i, m.trees[end])
            if i % print_every_n == 0 && verbosity > 0
                @info "iter $i" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    if String(device) == "gpu"
        GC.gc(true)
        CUDA.reclaim()
    end

    if return_logger
        return (m, logger)
    else
        return m
    end

end


"""
    fit_evotree(params::EvoTypes{L};
        x_train::AbstractMatrix, y_train::AbstractVector, w_train=nothing, offset_train=nothing,
        x_eval=nothing, y_eval=nothing, w_eval=nothing, offset_eval=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` and other optional kwargs. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` can be one of: 
    - [`EvoTreeRegressor`](@ref)
    - [`EvoTreeClassifier`](@ref)
    - [`EvoTreeCount`](@ref)
    - [`EvoTreeMLE`](@ref)

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
    - `:gaussian_mle`: Gaussian maximum log-likelihood. Adapted to `EvoTreeMLE` models with `loss = :gaussian_mle`. 
    - `:logistic_mle`: Logistic maximum log-likelihood. Adapted to `EvoTreeMLE` models with `loss = :logistic_mle`. 
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `fnames`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
- `return_logger::Bool = false`: if set to true (default), `fit_evotree` return a tuple `(m, logger)` where logger is a dict containing various tracking information.
- `device="cpu"`: Hardware device to use for computations. Can be either `"cpu"` or `"gpu"`. Following losses are not GPU supported at the moment`
    :l1`, `:quantile`, `:logistic_mle`.
"""
function fit_evotree(
    params::EvoTypes{L};
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train=nothing,
    offset_train=nothing,
    x_eval=nothing,
    y_eval=nothing,
    w_eval=nothing,
    offset_eval=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    fnames=nothing,
    return_logger=false,
    device="cpu"
) where {L}

    verbosity == 1 && @info params
    @assert string(device) ∈ ["cpu", "gpu"]
    _device = string(device) == "cpu" ? CPU : GPU

    m, cache = init(params, x_train, y_train, _device; fnames, w_train, offset_train)

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(x_eval) && !isnothing(y_eval)
    any_flag = !isnothing(metric) || !isnothing(x_eval) || !isnothing(y_eval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, `metric`, `x_eval` and `y_eval` must all be provided."
    end
    if logging_flag
        cb = CallBack(params, m, x_eval, y_eval, _device; w_eval, offset_eval, metric)
        logger = init_logger(; metric, maximise=is_maximise(cb.feval), early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    for i = 1:params.nrounds
        grow_evotree!(m, cache, params, _device)
        if !isnothing(logger)
            cb(logger, i, m.trees[end])
            if i % print_every_n == 0 && verbosity > 0
                @info "iter $i" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    if _device <: GPU
        GC.gc(true)
        CUDA.reclaim()
    end

    if return_logger
        return (m, logger)
    else
        return m
    end
end
