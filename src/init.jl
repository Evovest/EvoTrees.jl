"""
    orient_matrix_target(y, nobs)

Bring a matrix target to the internal layout `(n_targets, nobs)`.

The public matrix `fit` API takes `y` as `(nobs, n_targets)`, matching `x_train`
`(nobs, nfeats)`. Gradients are stored `(K, nobs)`, so this permutes once at the
boundary. A matrix that is already `(n_targets, nobs)` (second dim equals `nobs`,
first does not) is left as-is. Vectors are returned unchanged.
"""
function orient_matrix_target(y::AbstractVector, nobs::Integer)
    length(y) == nobs || error(
        "`y` has length $(length(y)) but there are $nobs observations. They must match."
    )
    return y
end
function orient_matrix_target(y::AbstractMatrix, nobs::Integer)
    nrows, ncols = size(y)
    if nrows == nobs
        return permutedims(y)
    elseif ncols == nobs
        return y
    else
        error(
            "`y` has size $(size(y)); one dimension must equal the number of observations ($nobs). " *
            "Pass `(nobs, n_targets)` to match `x_train`."
        )
    end
end

"""
    _init_target(::Type{L}, y_train, params, offset, ::Type{T})

Shared (device-agnostic) target/bias initialization: validates the target,
derives the output dimension `K`, the converted target `y` (host arrays;
device inits copy to their backend afterwards), and the initial bias `μ`.
Mutates `offset` in place into link space when provided. Single source of
truth for CPU (`src/init.jl`) and GPU (`ext/.../init.jl`) initialization.
"""
function _init_target(::Type{L}, y_train, params, offset, ::Type{T}) where {L,T}
    if (y_train isa AbstractMatrix) && !(L <: Union{GradientRegression, MLE2P, MAE, Quantile, Cred})
        error("Multi-target (matrix target) is supported for gradient-regression losses " *
              "(mse, logloss, poisson, gamma, tweedie), mae, quantile, the MLE losses " *
              "(gaussian_mle, logistic_mle), and credibility losses (cred_var, cred_std). " *
              "Got loss $(params.loss).")
    end
    if eltype(y_train) <: Real
        i = findfirst(!isfinite, y_train)
        isnothing(i) || error(
            "Target must be finite, got $(y_train[i]) at index $i. A non-finite target " *
            "propagates through the initial bias into every gradient and leaf, leaving a " *
            "model that predicts NaN everywhere."
        )
    end
    target_levels = nothing
    target_isordered = false
    if L == LogLoss
        @assert eltype(y_train) <: Real && minimum(y_train) >= 0 && maximum(y_train) <= 1
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[logit(mean(y))]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[logit(mean(view(y, k, :))) for k in 1:K]
        end
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        @assert eltype(y_train) <: Real
        if L == Gamma
            ymin = minimum(y_train)
            ymin <= 0 && error(
                "Gamma regression requires a strictly positive target, got a minimum of $ymin. " *
                "The gamma deviance is undefined at 0."
            )
        elseif L == Tweedie
            ymin = minimum(y_train)
            ymin < 0 && error(
                "Tweedie regression requires a non-negative target, got a minimum of $ymin. " *
                "The tweedie deviance is undefined below 0."
            )
        elseif L == Poisson
            ymin = minimum(y_train)
            ymin < 0 && error(
                "Poisson regression requires a non-negative target, got a minimum of $ymin. " *
                "The poisson deviance is undefined below 0."
            )
        end
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[log(mean(y))]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[log(mean(view(y, k, :))) for k in 1:K]
        end
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == MLogLoss
        if eltype(y_train) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y_train)
            target_isordered = isordered(y_train)
            y = UInt32.(CategoricalArrays.levelcode.(y_train))
        elseif eltype(y_train) <: Integer || eltype(y_train) <: Bool || eltype(y_train) <: String || eltype(y_train) <: Char
            yc = categorical(y_train, levels=sort(unique(y_train)), ordered=false)
            target_levels = CategoricalArrays.levels(yc)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        else
            error("Invalid target eltype: $(eltype(y_train))")
        end
        K = length(target_levels)
        K < 2 && error(
            "Classification requires a target with at least 2 levels, got $K: " *
            "$(string.(target_levels)). A single-class problem is not meaningful."
        )
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 2
            y = T.(y_train)
            μ = [mean(y), log(std(y))]
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        else
            Y = size(y_train, 1)
            K = 2 * Y
            y = T.(y_train)
            μ = T[]
            for t in 1:Y
                yt = view(y, t, :)
                push!(μ, mean(yt), log(std(yt)))
            end
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        end
    elseif L == LogisticMLE
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 2
            y = T.(y_train)
            μ = [mean(y), log(std(y) * sqrt(3) / π)]
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        else
            Y = size(y_train, 1)
            K = 2 * Y
            y = T.(y_train)
            μ = T[]
            for t in 1:Y
                yt = view(y, t, :)
                push!(μ, mean(yt), log(std(yt) * sqrt(3) / π))
            end
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        end
    elseif L == MultiQuantile
        @assert eltype(y_train) <: Real
        K = length(params.alphas)
        y = T.(y_train)
        μ = T.(quantile.(Ref(y), params.alphas))
    elseif L <: Union{MAE,Quantile}
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[mean(y)]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[mean(view(y, k, :)) for k in 1:K]
        end
    elseif L <: Cred
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[mean(y)]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[mean(view(y, k, :)) for k in 1:K]
        end
    else
        @assert eltype(y_train) <: Real
        if L == LambdaRank
            # Scores are relative within a query, so a constant bias cancels.
            @assert minimum(y_train) >= 0 "`:lambdarank` requires non-negative graded relevance."
            K = 1
            y = T.(y_train)
            μ = T[0]
        elseif L == Pearson
            # A constant bias does not change a correlation within a group.
            y_train isa AbstractMatrix && error("`loss = :pearson` takes a single target vector, not a matrix.")
            K = 1
            y = T.(y_train)
            i = findfirst(!isfinite, y)
            isnothing(i) || error("`loss = :pearson` needs a target that is finite in Float32, got $(y_train[i]) at index $i.")
            μ = T[0]
        elseif L <: GradientRegression
            if y_train isa AbstractVector
                K = 1
                y = T.(y_train)
                μ = T[mean(y)]
            else
                K = size(y_train, 1)
                y = T.(y_train)
                μ = T[mean(view(y, k, :)) for k in 1:K]
            end
        else
            K = 1
            y = T.(y_train)
            μ = [mean(y)]
        end
    end
    μ = T.(μ)
    return K, y, μ, target_levels, target_isordered
end

"""
    check_ctrl(params, ctrl, L, group)

Throw if the controls, groups, learner and loss are not a valid combination.
"""
function check_ctrl(params::EvoTypes, ctrl, ::Type{L}, group) where {L}
    if !isnothing(ctrl) && !hasproperty(params, :ctrl_lambda)
        error("A control variable was given but $(typeof(params)) has no `ctrl_lambda` to weigh " *
              "it with. The decorrelation penalty is available on `EvoTreeRegressor` and `EvoTreeMLE`.")
    end
    lambda = hasproperty(params, :ctrl_lambda) ? params.ctrl_lambda : 0.0
    # the field can be reassigned after construction
    hasproperty(params, :ctrl_lambda) &&
        check_parameter(Float64, lambda, zero(Float64), floatmax(Float64), :ctrl_lambda)
    if lambda > 0 && isnothing(ctrl)
        error("`ctrl_lambda` is $lambda but no control variable was given. Pass `ctrl_name` " *
              "when fitting from a table, or `ctrl_train` alongside `x_train`, to `EvoTrees.fit`; " *
              "the MLJ interface has no way to pass one.")
    end
    within = hasproperty(params, :ctrl_within_group) && params.ctrl_within_group
    if within && !isnothing(ctrl) && isnothing(group)
        error("`ctrl_within_group` is set but no groups were given. Pass `group_name` when " *
              "fitting from a table, or `group_train` alongside `x_train`.")
    end
    # listed explicitly so a new subtype does not inherit the penalty
    if !isnothing(ctrl) && !(L in (MSE, LogLoss, Poisson, Gamma, Tweedie, GaussianMLE))
        error("The decorrelation penalty is available for :mse, :logloss, :poisson, :gamma, " *
              ":tweedie and :gaussian_mle, not for $(params.loss).")
    end
    return nothing
end

"""
    dcor_cache(params, ctrl::Controls, group, w)

A `SummedDcorCache` over the `k` controls, pooled or per group.
"""
function dcor_cache(params::EvoTypes, ctrl::Controls, group, w)
    wh = w isa Array ? w : Array(w)
    wbar = _ordered_sum(wh) / length(wh)
    within = hasproperty(params, :ctrl_within_group) && params.ctrl_within_group
    n = length(first(ctrl.cols))
    if within
        many = length(ctrl.cols) > 1
        caches = [GroupedDcorCache(c, group; w=wh, label=many ? l : nothing) for (c, l) in zip(ctrl.cols, ctrl.labels)]
        kept = falses(n)
        for c in caches, r in c.rows
            kept[r] .= true
        end
        rows = findall(kept)
        return SummedDcorCache(caches, ctrl.weights, zeros(n), rows, wbar)
    else
        caches = [DcorCache(c; wbar) for c in ctrl.cols]
        return SummedDcorCache(caches, ctrl.weights, zeros(n), Int[], wbar)
    end
end
dcor_cache(::EvoTypes, ::Nothing, group, w) = nothing
# a single control as `build_ctrl` returns it
dcor_cache(params::EvoTypes, ctrl::AbstractVector{<:Real}, group, w) =
    dcor_cache(params, Controls([Vector{Float64}(ctrl)], [1.0], ["the control"]), group, w)

"""
    build_ctrl(ctrl_raw, nobs, argname; col=nothing, eval=false)

Validate a control of length `nobs` and return it standardised, as a `Vector{Float64}` on the host.
"""
function build_ctrl(ctrl_raw, nobs::Int, argname::AbstractString; col=nothing, eval=false)
    label = isnothing(col) ? "`$argname`" : "column $col of `$argname`"
    why = eval ? "its dependence on the evaluation set is undefined" : "there is no dependence for the penalty to remove"
    nonmissingtype(eltype(ctrl_raw)) <: Real ||
        error("$label must hold real numbers, got elements of type $(eltype(ctrl_raw)).")
    Missing <: eltype(ctrl_raw) && any(ismissing, ctrl_raw) &&
        error("$label contains missing values. Replace them before passing it.")
    ctrl = Vector{Float64}(vec(ctrl_raw))
    length(ctrl) == nobs ||
        error("$label has length $(length(ctrl)) but there are $nobs observations.")
    length(ctrl) >= 4 ||
        error("$label needs at least 4 observations for a distance covariance, got $(length(ctrl)).")
    all(isfinite, ctrl) || error("$label contains a non-finite value.")
    # `extrema`, unlike `allequal`, sees `0.0` and `-0.0` as equal
    lo, hi = extrema(ctrl)
    lo < hi ||
        error("$label is constant, so $why.")
    # at most one value either side of a tie gives zero distance variance
    s = sort(ctrl)
    s[2] < s[end-1] ||
        error("$label has every value tied but at most one either side of them, so its distance " *
              "variance is zero and $why.")
    m, sd = _ordered_mean_std(ctrl)
    isfinite(sd) && sd > 0 ||
        error("$label has a standard deviation of $sd, which cannot be used to scale it. " *
              "Rescale the column before passing it.")
    ctrl .= (ctrl .- m) ./ sd
    return ctrl
end

"""
    build_ctrl_weights(ctrl_weights, k)

Validated weights of the `k` controls as a `Vector{Float64}`, ones by default.
"""
function build_ctrl_weights(ctrl_weights, k::Int)
    isnothing(ctrl_weights) && return ones(k)
    ctrl_weights isa AbstractVector && all(v -> v isa Real, ctrl_weights) ||
        error("`ctrl_weights` must be a vector of real numbers, one per control.")
    length(ctrl_weights) == k ||
        error("`ctrl_weights` has $(length(ctrl_weights)) entries but there are $k controls.")
    ω = Float64.(ctrl_weights)
    all(v -> isfinite(v) && v > 0, ω) ||
        error("`ctrl_weights` must be positive and finite, got $(ctrl_weights).")
    return ω
end

"""
    build_ctrls(ctrl_train, nobs, ctrl_weights; argname="ctrl_train", eval=false)

`Controls` from a vector of length `nobs` or a matrix of size `(nobs, k)`.
"""
function build_ctrls(ctrl_train, nobs::Int, ctrl_weights; argname="ctrl_train", eval=false)
    if ctrl_train isa AbstractMatrix
        size(ctrl_train, 1) == nobs ||
            error("`$argname` has $(size(ctrl_train, 1)) rows but there are $nobs observations. " *
                  "Pass one row per observation and one column per control.")
        size(ctrl_train, 2) >= 1 || error("`$argname` has no columns.")
        js = axes(ctrl_train, 2)
        cols = [build_ctrl(view(ctrl_train, :, j), nobs, argname; col=j, eval) for j in js]
        labels = ["column $j of `$argname`" for j in js]
    else
        cols = [build_ctrl(ctrl_train, nobs, argname; eval)]
        labels = ["`$argname`"]
    end
    return Controls(cols, build_ctrl_weights(ctrl_weights, length(cols)), labels)
end

"""
    build_ctrls(dtrain, ctrl_name, nobs, ctrl_weights)

`Controls` from the columns of `dtrain` named by `ctrl_name`, one name or a vector of names.
"""
function build_ctrls(dtrain, ctrl_name, nobs::Int, ctrl_weights)
    if ctrl_name isa AbstractVector
        names = Symbol.(ctrl_name)
        isempty(names) && error("`ctrl_name` is empty. Pass at least one column name.")
        allunique(names) || error("`ctrl_name` names the same column more than once.")
        cols = [build_ctrl(Tables.getcolumn(dtrain, nm), nobs, "ctrl_name"; col="`$nm`") for nm in names]
        labels = ["column `$nm` of `ctrl_name`" for nm in names]
    else
        cols = [build_ctrl(Tables.getcolumn(dtrain, Symbol(ctrl_name)), nobs, "ctrl_name")]
        labels = ["`ctrl_name`"]
    end
    return Controls(cols, build_ctrl_weights(ctrl_weights, length(cols)), labels)
end

function init_core(params::EvoTypes, ::Type{CPU}, data, feature_names, y_train, w, offset, group=nothing, ctrl=nothing)
    # the config can be mutated after construction, as MLJ tuning does
    check_args(params)

    # binarize data into quantiles
    rng = Xoshiro(params.seed)

    edges, featbins, feattypes = get_edges(data; feature_names, nbins=params.nbins, rng)
    x_bin = binarize(data; feature_names, edges)
    x_bin_T = permutedims(x_bin)
    nobs, nfeats = size(x_bin)

    T = Float32
    L = _loss2type_dict[params.loss]

    K, y, μ, target_levels, target_isordered = _init_target(L, y_train, params, offset, T)
    check_ctrl(params, ctrl, L, group)
    ctrl = dcor_cache(params, ctrl, group, w)

    # force a neutral/zero bias when offset is specified
    !isnothing(offset) && (μ .= 0)
    @assert (size(y, ndims(y)) == length(w) && minimum(w) > 0)

    # initialize preds
    pred = zeros(T, K, nobs)
    pred .= μ
    !isnothing(offset) && (pred .+= offset')

    # initialize gradients
    ∇ = zeros(T, 2 * K + 1, nobs)
    ∇[end, :] .= w

    # initialize indexes
    mask_cond = zeros(UInt8, nobs)
    is = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    # `h∇` is indexed by splittable node; `TrainNode` also exists for leaves (∑ / pred).
    n_hist = 2^params.max_depth - 1
    n_nodes = 2^(params.max_depth + 1) - 1
    nbins = params.nbins
    h∇ = zeros(Float64, 2 * K + 1, nbins, nfeats, n_hist)
    nodes = [TrainNode(zero(Float64), view(is, 1:0), zeros(Float64, 2 * K + 1), zeros(Float64, 2 * K + 1), zeros(Float64, 2 * K + 1), view(h∇, :, :, :, min(n, n_hist)), zeros(nbins, nfeats)) for n = 1:n_nodes]
    m = EvoTree{L,K}(L, K, μ, info)

    # build cache
    Y = typeof(y)
    N = typeof(first(nodes))
    H = typeof(h∇)
    G = typeof(group)
    cache = CacheBaseCPU{Y,N,H,G}(
        rng,
        K,
        x_bin,
        x_bin_T,
        y,
        w,
        pred,
        nodes,
        mask_cond,
        is,
        left,
        right,
        js,
        ∇,
        h∇,
        feature_names,
        featbins,
        feattypes,
        monotone_constraints,
        group,
        ctrl,
    )
    return m, cache
end

"""
    init(
        params::EvoTypes,
        dtrain,
        device::Type{<:Device}=CPU;
        target_name,
        feature_names=nothing,
        weight_name=nothing,
        offset_name=nothing,
        group_name=nothing,
        ctrl_name=nothing,
        ctrl_weights=nothing
    )

Initialise EvoTree. `group_name`, `ctrl_name` and `ctrl_weights` are as in `EvoTrees.fit`.
"""
function init(
    params::EvoTypes,
    dtrain,
    device::Type{<:Device}=CPU;
    target_name,
    feature_names=nothing,
    weight_name=nothing,
    offset_name=nothing,
    group_name=nothing,
    ctrl_name=nothing,
    ctrl_weights=nothing
)

    # set feature_names
    schema = Tables.schema(dtrain)
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _group_name = isnothing(group_name) ? Symbol("") : Symbol(group_name)
    _ctrl_names = isnothing(ctrl_name) ? Symbol[] : ctrl_name isa AbstractVector ? Symbol.(ctrl_name) : [Symbol(ctrl_name)]
    _target_names = target_name isa AbstractVector ? Symbol.(target_name) : [Symbol(target_name)]
    if isnothing(feature_names)
        feature_names = Symbol[]
        for i in eachindex(schema.names)
            if schema.types[i] <: Union{Real,CategoricalValue}
                push!(feature_names, schema.names[i])
            end
        end
        feature_names = setdiff(feature_names, union(_target_names, [_weight_name], [_offset_name], [_group_name], _ctrl_names))
    else
        isa(feature_names, String) ? feature_names = [feature_names] : nothing
        feature_names = Symbol.(feature_names)
        @assert isa(feature_names, Vector{Symbol})
        @assert all(feature_names .∈ Ref(schema.names))
        for name in feature_names
            @assert schema.types[findfirst(name .== schema.names)] <: Union{Real,CategoricalValue}
        end
    end

    T = Float32
    nobs = length(Tables.getcolumn(dtrain, 1))
    y_train = length(_target_names) == 1 ?
        Tables.getcolumn(dtrain, _target_names[1]) :
        permutedims(reduce(hcat, [Tables.getcolumn(dtrain, t) for t in _target_names]))
    V = device_array_type(device)
    w = isnothing(weight_name) ? device_ones(device, T, nobs) : V{T}(Tables.getcolumn(dtrain, _weight_name))
    offset = isnothing(offset_name) ? nothing : V{T}(Tables.getcolumn(dtrain, _offset_name))
    group = isnothing(group_name) ? nothing : build_group_index(Tables.getcolumn(dtrain, _group_name), nobs, "group_name")
    isnothing(ctrl_name) && !isnothing(ctrl_weights) &&
        error("`ctrl_weights` was given without a control variable. Pass `ctrl_name` with it.")
    ctrl = isnothing(ctrl_name) ? nothing : build_ctrls(dtrain, ctrl_name, nobs, ctrl_weights)

    m, cache = init_core(params, device, dtrain, feature_names, y_train, w, offset, group, ctrl)

    m.info[:target_names] = _target_names
    m.info[:group_name] = isnothing(group_name) ? nothing : _group_name
    m.info[:ctrl_names] = _ctrl_names

    return m, cache
end

# This should be different on CPUs and GPUs
device_ones(::Type{<:CPU}, ::Type{T}, n::Int) where {T} = ones(T, n)
device_array_type(::Type{<:CPU}) = Array

"""
    init(
        params::EvoTypes,
        x_train::AbstractMatrix,
        y_train::AbstractVecOrMat,
        device::Type{<:Device}=CPU;
        feature_names=nothing,
        w_train=nothing,
        offset_train=nothing,
        group_train=nothing,
        ctrl_train=nothing,
        ctrl_weights=nothing
    )

Initialise EvoTree. `group_train`, `ctrl_train` and `ctrl_weights` are as in `EvoTrees.fit`.
"""
function init(
    params::EvoTypes,
    x_train::AbstractMatrix,
    y_train::AbstractVecOrMat,
    device::Type{<:Device}=CPU;
    feature_names=nothing,
    w_train=nothing,
    offset_train=nothing,
    group_train=nothing,
    ctrl_train=nothing,
    ctrl_weights=nothing
)

    # initialize model and cache
    feature_names = isnothing(feature_names) ? [Symbol("feat_$i") for i in axes(x_train, 2)] : Symbol.(feature_names)
    @assert length(feature_names) == size(x_train, 2)

    T = Float32
    nobs = size(x_train, 1)
    y_train = orient_matrix_target(y_train, nobs)
    V = device_array_type(device)
    w = isnothing(w_train) ? device_ones(device, T, nobs) : V{T}(w_train)
    offset = isnothing(offset_train) ? nothing : V{T}(offset_train)
    group = isnothing(group_train) ? nothing : build_group_index(group_train, nobs, "group_train")

    isnothing(ctrl_train) && !isnothing(ctrl_weights) &&
        error("`ctrl_weights` was given without a control variable. Pass `ctrl_train` with it.")
    ctrl = isnothing(ctrl_train) ? nothing : build_ctrls(ctrl_train, nobs, ctrl_weights)
    m, cache = init_core(params, device, x_train, feature_names, y_train, w, offset, group, ctrl)

    return m, cache
end
