"""
    grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTypes) where {L,K}

Given a instantiate
"""
function grow_evotree!(m::EvoTree{L,K}, cache::CacheCPU, params::EvoTypes) where {L,K}

    # compute gradients
    update_grads!(cache.∇, cache.pred, cache.y, L, params)
    # subsample rows
    cache.nodes[1].is = subsample(cache.left, cache.is, cache.mask_cond, params.rowsample, params.rng)
    # subsample cols
    sample!(params.rng, UInt32(1):UInt32(length(cache.feattypes)), cache.js, replace=false, ordered=true)

    # instantiate a tree then grow it
    tree = Tree{L,K}(params.max_depth)
    grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
    grow!(
        tree,
        cache.nodes,
        params,
        cache.∇,
        cache.js,
        cache.is,
        cache.left,
        cache.right,
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints
    )
    push!(m.trees, tree)
    predict!(cache.pred, tree, cache.x_bin, cache.feattypes)
    m.info[:nrounds] += 1
    return nothing
end

# grow a single tree
function grow_tree!(
    tree::Tree{L,K},
    nodes::Vector{N},
    params::EvoTypes,
    ∇,
    js,
    is,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,N}

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ = sum(view(∇, nodes[1].is))
    nodes[1].gain = get_gain(L, params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        # pred leafs if max depth is reached
        if depth == params.max_depth
            for n ∈ n_current
                if L <: Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            end
        else
            # look for best split for each node
            @threads for n ∈ n_current[1:2:end]
                update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
            end
            @threads for n ∈ n_current[2:2:end]
                if n % 2 == 0
                    @views nodes[n].h[:, js] .= nodes[n>>1].h[:, js] .- nodes[n+1].h[:, js]
                else
                    @views nodes[n].h[:, js] .= nodes[n>>1].h[:, js] .- nodes[n-1].h[:, js]
                end
            end
            sort!(n_current)
            @threads for n ∈ n_current
                best_gain, best_feat, best_bin = get_best_split(L, nodes[n], js, params, feattypes, monotone_constraints)
                if best_bin != 0
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.split[n] = true
                end
            end

            for n ∈ n_current
                if tree.split[n]

                    best_feat = tree.feat[n]
                    best_bin = tree.cond_bin[n]

                    _left, _right = split_set!(
                        nodes[n].is,
                        is,
                        left,
                        right,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )
                    offset += length(nodes[n].is)

                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ = nodes[n].hL[best_bin, best_feat]
                    nodes[n<<1+1].∑ = nodes[n].hR[best_bin, best_feat]
                    nodes[n<<1].gain = get_gain(L, params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(L, params, nodes[n<<1+1].∑)

                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
                else
                    if L <: Quantile
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                    else
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                    end
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
    params::EvoTypes,
    ∇,
    js,
    is,
    left,
    right,
    x_bin,
    feattypes::Vector{Bool},
    monotone_constraints
) where {L,K,N}

    # initialize
    n_current = [1]
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= dropdims(sum(Float64, view(∇, :, nodes[1].is), dims=2), dims=2)
    nodes[1].gain = get_gain(L, params, nodes[1].∑)

    # grow while there are remaining active nodes
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        n_next = Int[]

        if depth == params.max_depth
            for n in n_current
                if L <: Quantile
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                else
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                end
            end
        else
            # look for best split for each node
            @threads for n ∈ n_current[1:2:end]
                update_hist!(L, nodes[n].h, ∇, x_bin, nodes[n].is, js)
            end
            @threads for n ∈ n_current[2:2:end]
                if n % 2 == 0
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n+1].h[:, :, js]
                else
                    @views nodes[n].h[:, :, js] .= nodes[n>>1].h[:, :, js] .- nodes[n-1].h[:, :, js]
                end
            end
            sort!(n_current)
            @threads for n ∈ n_current
                update_gains!(L, nodes[n], js, params, feattypes, monotone_constraints)
            end

            # initialize gains for node 1 in which all gains of a given depth will be accumulated
            if depth > 1
                view(nodes[1].gains, :, js) .= 0
            end
            gain = 0
            # update gains based on the aggregation of all nodes of a given depth. One gains matrix per depth (vs one per node in binary trees).
            for n ∈ n_current
                if n > 1 # accumulate gains in node 1
                    @views nodes[1].gains[:, js] .+= nodes[n].gains[:, js]
                end
                gain += nodes[n].gain
            end
            for n ∈ n_current
                if n > 1
                    @views nodes[1].gains[:, js] .*= nodes[n].gains[:, js] .> 0 #mask ignore gains if any node isn't eligible (too small per leaf weight)
                end
            end
            # find best split
            best = findmax(view(nodes[1].gains, :, js))
            best_gain = best[1]
            best_bin = best[2][1]
            best_feat = js[best[2][2]]
            if best_gain > gain + params.gamma
                for n in n_current
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.split[n] = best_bin != 0

                    _left, _right = split_set!(
                        nodes[n].is,
                        is,
                        left,
                        right,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )
                    offset += length(nodes[n].is)

                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[:, best_bin, best_feat]
                    nodes[n<<1+1].∑ .= nodes[n].hR[:, best_bin, best_feat]
                    nodes[n<<1].gain = get_gain(L, params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(L, params, nodes[n<<1+1].∑)

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
                    if L <: Quantile
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params, ∇, nodes[n].is)
                    else
                        pred_leaf_cpu!(tree.pred, n, nodes[n].∑, L, params)
                    end
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over current nodes for a given depth
    return nothing
end

# A no-op on the CPU, but on the GPU we perform garbage collection
post_fit_gc(::Type{<:CPU}) = nothing

"""
    fit(
        params::EvoTypes, 
        dtrain;
        target_name,
        feature_names=nothing,
        weight_name=nothing,
        offset_name=nothing,
        deval=nothing,
        print_every_n=9999,
        verbosity=1
)

Main training function. Performs model fitting given configuration `params`, `dtrain`, `target_name` and other optional kwargs. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` can be one of: 
    - [`EvoTreeRegressor`](@ref)
    - [`EvoTreeClassifier`](@ref)
    - [`EvoTreeCount`](@ref)
    - [`EvoTreeMLE`](@ref)
- `dtrain`: A Tables compatible training data (named tuples, DataFrame...) containing features and target variables. 

# Keyword arguments

- `target_name`: name of the target variable. 
- `feature_names = nothing`: the names `dtrain` variables to use as features. If not provided, it deafults to all variables that aren't one of `target`, `weight` or `offset``.
- `weight_name = nothing`: name of the variable containing weights. If `nothing`, common weights on one will be used.
- `offset_name = nothing`: name of the offset variable.
- `deval`: A Tables compatible evaluation data containing features and target variables. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
"""
function fit(
    params::EvoTypes,
    dtrain;
    target_name,
    feature_names=nothing,
    weight_name=nothing,
    offset_name=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1,
)

    @assert Tables.istable(dtrain) "fit(params, dtrain) only accepts Tables compatible input for `dtrain` (ex: named tuples, DataFrames...)"
    dtrain = Tables.columntable(dtrain)
    _device = params.device == :gpu ? GPU : CPU
    m, cache = init(params, dtrain, _device; target_name, feature_names, weight_name, offset_name)

    # initialize callback and logger if deval is provided
    if !isnothing(deval)
        deval = Tables.columntable(deval)
        cb = CallBack(params, m, deval, _device; target_name, weight_name, offset_name)
        logger = init_logger(; metric=params.metric, maximise=is_maximise(cb.feval), params.early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    for i = 1:params.nrounds
        grow_evotree!(m, cache, params)
        if !isnothing(logger)
            cb(logger, i, m.trees[end])
            if i % print_every_n == 0 && verbosity > 0
                @info "iter $i" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    post_fit_gc(_device)
    m.info[:logger] = logger

    return m

end

"""
    fit(
        params::EvoTypes{L};
        x_train::AbstractMatrix, 
        y_train::AbstractVector, 
        w_train=nothing, 
        offset_train=nothing,
        x_eval=nothing, 
        y_eval=nothing, 
        w_eval=nothing, 
        offset_eval=nothing,
        feature_names=nothing,
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
- `feature_names = nothing`: the names of the `x_train` features. If provided, should be a vector of string with `length(feature_names) = size(x_train, 2)`.
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
"""
function fit(
    params::EvoTypes;
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train=nothing,
    offset_train=nothing,
    x_eval=nothing,
    y_eval=nothing,
    w_eval=nothing,
    offset_eval=nothing,
    feature_names=nothing,
    print_every_n=9999,
    verbosity=1
)

    _device = params.device == :gpu ? GPU : CPU
    m, cache = init(params, x_train, y_train, _device; feature_names, w_train, offset_train)

    # initialize callback and logger if tracking eval data
    metric = params.metric
    logging_flag = !isnothing(x_eval) && !isnothing(y_eval)
    any_flag = !isnothing(x_eval) || !isnothing(y_eval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, both `x_eval` and `y_eval` must be provided."
    end
    if logging_flag
        cb = CallBack(params, m, x_eval, y_eval, _device; w_eval, offset_eval)
        logger = init_logger(; metric=params.metric, maximise=is_maximise(cb.feval), params.early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    for i = 1:params.nrounds
        grow_evotree!(m, cache, params)
        if !isnothing(logger)
            cb(logger, i, m.trees[end])
            if i % print_every_n == 0 && verbosity > 0
                @info "iter $i" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    post_fit_gc(_device)
    m.info[:logger] = logger

    return m

end



"""
    fit_evotree(
        params::EvoTypes, 
        dtrain;
        target_name,
        feature_names=nothing,
        weight_name=nothing,
        offset_name=nothing,
        deval=nothing,
        print_every_n=9999,
        verbosity=1
)

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
- `feature_names = nothing`: the names `dtrain` variables to use as features. If not provided, it deafults to all variables that aren't one of `target`, `weight` or `offset``.
- `weight_name = nothing`: name of the variable containing weights. If `nothing`, common weights on one will be used.
- `offset_name = nothing`: name of the offset variable.
- `deval`: A Tables compatible evaluation data containing features and target variables. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
"""
function fit_evotree(
    params::EvoTypes,
    dtrain;
    target_name,
    feature_names=nothing,
    weight_name=nothing,
    offset_name=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1,
    kwargs...
)

    Base.depwarn(
        "`fit_evotree` has been deprecated, use `fit` instead. 
        Following kwargs are no longer supported in `fit_evotree`: `metric`, `return_logger`, `early_stopping_rounds` and `device`.
        See docs on how to get those functionalities through the model builder (ex: `EvoTreeRegressor`) and `fit`.",
        :fit_evotree
    )

    return fit(params, dtrain;
        target_name,
        feature_names,
        weight_name,
        offset_name,
        deval,
        print_every_n,
        verbosity)

end

"""
    fit_evotree(
        params::EvoTypes{L};
        x_train::AbstractMatrix, 
        y_train::AbstractVector, 
        w_train=nothing, 
        offset_train=nothing,
        x_eval=nothing, 
        y_eval=nothing, 
        w_eval=nothing, 
        offset_eval=nothing,
        feature_names=nothing,
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
- `feature_names = nothing`: the names of the `x_train` features. If provided, should be a vector of string with `length(feature_names) = size(x_train, 2)`.
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
"""
function fit_evotree(
    params::EvoTypes;
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    w_train=nothing,
    offset_train=nothing,
    x_eval=nothing,
    y_eval=nothing,
    w_eval=nothing,
    offset_eval=nothing,
    feature_names=nothing,
    print_every_n=9999,
    verbosity=1,
    kwargs...
)

    Base.depwarn(
        "`fit_evotree` has been deprecated, use `fit` instead. 
        Following kwargs are no longer supported in `fit_evotree`: `metric`, `return_logger`, `early_stopping_rounds` and `device`.
        See docs on how to get those functionalities through the model builder (ex: `EvoTreeRegressor`) and `fit`.",
        :fit_evotree
    )

    return fit(
        params;
        x_train,
        y_train,
        w_train,
        offset_train,
        x_eval,
        y_eval,
        w_eval,
        offset_eval,
        print_every_n,
        verbosity,
        feature_names
    )

    return m

end
