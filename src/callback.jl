_raw_level(x) = x isa CategoricalValue ? CategoricalArrays.unwrap(x) : x

"""
    eval_levelcode(y_eval, target_levels)

Encode `y_eval` against the class levels the model was trained on, rather than against the
levels present in `y_eval` itself. An eval set that is missing a training class, or that
orders its own levels differently, would otherwise be scored against the wrong prediction
columns.
"""
function eval_levelcode(y_eval, target_levels)
    idx = indexin(y_eval, target_levels)
    if any(isnothing, idx)
        unseen = unique(_raw_level.(y_eval[findall(isnothing, idx)]))
        error("`y_eval` contains levels absent from `y_train`: $(unseen). " *
              "Training levels are $(_raw_level.(target_levels)).")
    end
    return UInt32.(idx)
end

# Mirrors the assertion training makes on its own inputs in `src/init.jl`. Eval data was
# accepted unchecked, so a short weight vector silently scored a subset of the eval set and a
# weight vector summing to zero produced a NaN or Inf metric.
function check_eval_data(y, w, nobs)
    size(y, ndims(y)) == nobs || error(
        "`y_eval` has $(size(y, ndims(y))) observations but the evaluation features have " *
        "$(nobs). They must match."
    )
    length(w) == nobs || error(
        "`w_eval` has length $(length(w)) but the evaluation features have $(nobs) " *
        "observations. Each row needs exactly one weight."
    )
    minimum(w) > 0 || error("`w_eval` must be strictly positive.")
    return nothing
end

struct CallBack{B,P,Y,C,D,K,R}
    feval::Function
    x_bin::B
    p::P
    y::Y
    w::C
    eval::C
    feattypes::D
    metric_kwargs::K
    ctrl::R
end

"""
    EvalCtrl

The eval set's controls, for the penalised eval metric of a fit with a decorrelation penalty: one
cache per control, over the whole eval set or within each eval group, built on the eval controls
centred and scaled on the eval set itself, as the training controls are on theirs. The penalty is
then priced the same way on both sets, an eval date has a scaling of its own as a training date
does, and the metric depends only on the model and the eval data.

When `penalized`, the loss's own metric becomes the penalised objective on the eval set, in the
metric's units: under `:mse` the training objective per unit weight is `mse + ctrl_lambda * dcov2`,
and the eval value is the same expression on the eval predictions. `dep` holds each control's
dependence of the last round, averaged over the penalised outputs and before `ctrl_lambda` and the
control weights, so it compares across weights.
"""
struct EvalCtrl{L,C<:AbstractDcorCache}
    caches::Vector{C}
    weights::Vector{Float64}
    lambda::Float64
    penalized::Bool
    sign::Float64
    n::Int
    rows::Vector{Int}
    x::Vector{Float64}
    a::Vector{Float64}
    dep::Vector{Float64}
end

# The metric each loss's penalty can be priced in: a fixed multiple of its training objective per
# unit weight. Any other metric, a root, absolute, rank or correlation scale, has no exchange rate
# for `ctrl_lambda`, so it is reported as it stands, with the dependence logged beside it.
_own_metric(::Type{MSE}) = mse
_own_metric(::Type{LogLoss}) = logloss
_own_metric(::Type{Poisson}) = poisson
_own_metric(::Type{Gamma}) = gamma
_own_metric(::Type{Tweedie}) = tweedie
_own_metric(::Type{GaussianMLE}) = gaussian_mle
_own_metric(::Type) = nothing

# The penalty's gradient is `ctrl_lambda * W * g * h / 2w`, and `g` sees the prediction only through
# its ranks, so where `h / 2w` is the derivative of one increasing function of the prediction, the
# penalty is exactly `ctrl_lambda * W * dcov2` of that function. It is the prediction under `:mse`,
# half the probability under `:logloss` and half the mean under `:poisson`, whose deviance metric is
# twice its objective. Under `:gamma` and `:tweedie` the weight also depends on the target and is
# that derivative in expectation, so the term is exact at a calibrated fit; under `:gaussian_mle`
# the location's weight `1 / 2 scale^2` is taken at the fitted scale, exact when the scale is
# constant, and the metric is a log-likelihood, maximised, so the term is subtracted.
_dep_transform(::Type{<:Union{MSE,Gamma,GaussianMLE}}, p) = p
_dep_transform(::Type{LogLoss}, p) = sigmoid(p)
_dep_transform(::Type{Poisson}, p) = exp(p)
_dep_transform(::Type{Tweedie}, p) = exp((2 - 1.5) * p)   # rho = 1.5, as in the loss and its metric
_dep_coef(::Type{<:Union{MSE,Poisson,Gamma}}) = 1.0
_dep_coef(::Type{<:Union{LogLoss,GaussianMLE}}) = 0.5
_dep_coef(::Type{Tweedie}) = 1 / (2 - 1.5)

function eval_ctrl(params::EvoTypes, ::Type{L}, K::Int, ctrl::Controls, group, feval, n::Int) where {L}
    within = hasproperty(params, :ctrl_within_group) && params.ctrl_within_group
    if within
        isnothing(group) && error(
            "`ctrl_within_group` is set but the eval set has no groups. Pass `eval_group_name` when " *
            "fitting from a table, or `group_eval` alongside `x_eval`.")
        caches = [GroupedDcorCache(c, group; label=l) for (c, l) in zip(ctrl.cols, ctrl.labels)]
    else
        caches = [DcorCache(c) for c in ctrl.cols]
    end
    lambda = hasproperty(params, :ctrl_lambda) ? params.ctrl_lambda : 0.0
    penalized = lambda > 0 && feval === _own_metric(L)
    return EvalCtrl{L,eltype(caches)}(caches, ctrl.weights, lambda, penalized,
        is_maximise(feval) ? -1.0 : 1.0, n, collect(ctrl_rows(L, K)), zeros(n), zeros(n),
        zeros(length(caches)))
end

# one control's dependence on a prediction row, and the same with each row weighted by `a`
# (`:gaussian_mle`, where it is the location's curvature): over the whole eval set, or within
# groups as the size-weighted sum `sum_g n_g / n * dcov2_g`, a skipped group adding nothing
function _dependence(dc::DcorCache, x, a, n)
    d = dcov2_value!(dc, x)
    return d, isnothing(a) ? d : d * _ordered_sum(a) / length(a)
end
function _dependence(gc::GroupedDcorCache, x, a, n)
    dg = zeros(length(gc.caches))
    dag = zeros(length(gc.caches))
    @threads for k in eachindex(gc.caches)
        rows = gc.rows[k]
        dg[k] = length(rows) / n * dcov2_value!(gc.caches[k], view(x, rows))
        dag[k] = isnothing(a) ? dg[k] : dg[k] * _ordered_sum(view(a, rows)) / length(rows)
    end
    return _ordered_sum(dg), _ordered_sum(dag)
end

"""
    eval_penalty!(ec::EvalCtrl, p)

Each control's dependence on the eval predictions `p`, and the penalty term in the metric's units:
`sign * ctrl_lambda * coef * mean over penalised outputs of sum_c weight_c * dependence_c`. The
predictions are brought to the host first, as the training penalty does.
"""
function eval_penalty!(ec::EvalCtrl{L}, p::AbstractMatrix) where {L}
    ph = p isa Array ? p : Array(p)
    fill!(ec.dep, 0.0)
    term = 0.0
    for k in ec.rows
        @inbounds for i in eachindex(ec.x)
            ec.x[i] = _dep_transform(L, Float64(ph[k, i]))
        end
        a = nothing
        if L <: GaussianMLE
            @inbounds for i in eachindex(ec.a)
                ec.a[i] = exp(-2 * Float64(ph[k+1, i]))
            end
            a = ec.a
        end
        for c in eachindex(ec.caches)
            d, da = _dependence(ec.caches[c], ec.x, a, ec.n)
            ec.dep[c] += d
            term += ec.weights[c] * da
        end
    end
    ec.dep ./= length(ec.rows)
    return ec.dep, ec.sign * ec.lambda * _dep_coef(L) * term / length(ec.rows)
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    deval,
    device::Type{<:Device};
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    group_name=nothing,
    ctrl_name=nothing,
    ctrl_weights=nothing,
    x_bin=nothing) where {L,K}

    T = Float32
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_names = target_name isa AbstractVector ? Symbol.(target_name) : [Symbol(target_name)]

    if isnothing(x_bin)
        x_bin = binarize(device, deval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    end
    nobs = length(Tables.getcolumn(deval, 1))
    p = zeros(T, K, nobs)
    p .= m.bias

    y_eval = length(_target_names) == 1 ?
        Tables.getcolumn(deval, _target_names[1]) :
        permutedims(reduce(hcat, [Tables.getcolumn(deval, t) for t in _target_names]))

    if L == MLogLoss
        y = eval_levelcode(y_eval, m.info[:target_levels])
    else
        y = T.(y_eval)
    end
    feval = metric_dict[params.metric]
    V = device_array_type(device)
    w = isnothing(weight_name) ? device_ones(device, T, nobs) : V{T}(Tables.getcolumn(deval, _weight_name))
    check_eval_data(y, w, nobs)
    metric_kwargs = hasproperty(params, :alpha) ? (alpha=T(params.alpha),) : (;)
    if params.metric == :multiquantile
        alphas_eval = T.(params.alphas)
        device <: GPU && (alphas_eval = V{T}(alphas_eval))
        metric_kwargs = (alphas=alphas_eval,)
    end
    group_eval = nothing
    if !isnothing(group_name)
        group_eval = build_group_index(Tables.getcolumn(deval, Symbol(group_name)), nobs, "group_name")
        metric_kwargs = merge(metric_kwargs, (group=group_eval,))
    end
    hasproperty(params, :ndcg_k) && (metric_kwargs = merge(metric_kwargs, (ndcg_k=params.ndcg_k,)))

    # the eval set carries the same control columns as the training set
    ctrl = nothing
    if !isnothing(ctrl_name)
        names = ctrl_name isa AbstractVector ? Symbol.(ctrl_name) : [Symbol(ctrl_name)]
        for nm in names
            nm in Tables.columnnames(deval) || error(
                "`deval` has no column `$nm`. The eval set needs the same control columns as the " *
                "training set, for the eval metric to price the penalty.")
        end
        cols = [build_ctrl(Tables.getcolumn(deval, nm), nobs, "deval"; col="`$nm`", eval=true) for nm in names]
        ctrl = eval_ctrl(params, L, K, Controls(cols, ctrl_weights, ["column `$nm` of `deval`" for nm in names]),
            group_eval, feval, nobs)
    end

    offset = !isnothing(offset_name) ? T.(Tables.getcolumn(deval, _offset_name)) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && unconstrain_mle_scale!(offset)
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]), metric_kwargs, ctrl)
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    x_eval::AbstractMatrix,
    y_eval,
    device::Type{<:Device};
    w_eval=nothing,
    offset_eval=nothing,
    group_eval=nothing,
    ctrl_eval=nothing,
    ctrl_weights=nothing,
    x_bin=nothing) where {L,K}

    T = Float32
    nobs = size(x_eval, 1)
    if isnothing(x_bin)
        x_bin = binarize(device, x_eval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    end
    p = zeros(T, K, nobs)
    p .= m.bias
    y_eval = orient_matrix_target(y_eval, nobs)

    if L == MLogLoss
        y = eval_levelcode(y_eval, m.info[:target_levels])
    else
        y = T.(y_eval)
    end
    feval = metric_dict[params.metric]
    V = device_array_type(device)
    w = isnothing(w_eval) ? device_ones(device, T, nobs) : V{T}(w_eval)
    check_eval_data(y, w, nobs)
    metric_kwargs = hasproperty(params, :alpha) ? (alpha=T(params.alpha),) : (;)
    if params.metric == :multiquantile
        alphas_eval = T.(params.alphas)
        device <: GPU && (alphas_eval = V{T}(alphas_eval))
        metric_kwargs = (alphas=alphas_eval,)
    end
    gi = isnothing(group_eval) ? nothing : build_group_index(group_eval, nobs, "group_eval")
    isnothing(gi) || (metric_kwargs = merge(metric_kwargs, (group=gi,)))
    hasproperty(params, :ndcg_k) && (metric_kwargs = merge(metric_kwargs, (ndcg_k=params.ndcg_k,)))

    ctrl = nothing
    if !isnothing(ctrl_eval)
        ce = build_ctrls(ctrl_eval, nobs, nothing; argname="ctrl_eval", eval=true)
        length(ce.cols) == length(ctrl_weights) || error(
            "`ctrl_eval` has $(length(ce.cols)) controls but `ctrl_train` has $(length(ctrl_weights)). " *
            "Pass the same controls for both sets, in the same order.")
        ctrl = eval_ctrl(params, L, K, Controls(ce.cols, ctrl_weights, ce.labels), gi, feval, nobs)
    end

    offset = !isnothing(offset_eval) ? T.(offset_eval) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && unconstrain_mle_scale!(offset)
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]), metric_kwargs, ctrl)
end

function (cb::CallBack)(logger, iter)
    metric = cb.feval(cb.p, cb.y, cb.w, cb.eval; cb.metric_kwargs...)
    if isnothing(cb.ctrl)
        update_logger!(logger, iter, metric)
    else
        dep, term = eval_penalty!(cb.ctrl, cb.p)
        value = cb.ctrl.penalized ? metric + term : metric
        update_logger!(logger, iter, value; base=metric, dep=copy(dep))
    end
    return nothing
end

# `trees` is every tree added by the round being logged. With `bagging_size > 1` a round adds
# that many, each scaled by `1 / bagging_size`, so accumulating only one leaves the eval
# prediction at a fraction of the model the round actually produced.
function (cb::CallBack)(logger, iter, trees)
    for tree in trees
        predict!(cb.p, tree, cb.x_bin, cb.feattypes)
    end
    return cb(logger, iter)
end

# With controls the logger also keeps, every round, the base metric and each control's dependence,
# and whether `:metrics`, which early stopping reads, carries the penalty.
function init_logger(; metric, maximise, early_stopping_rounds, early_stopping_tolerance=0.0, ctrl=nothing)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :early_stopping_tolerance => early_stopping_tolerance,
        :nrounds => 0,
        :iter => Int[],
        :metrics => Float64[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    if !isnothing(ctrl)
        logger[:base_metrics] = Float64[]
        logger[:ctrl_dependence] = Vector{Float64}[]
        logger[:penalized] = ctrl.penalized
    end
    return logger
end

function update_logger!(logger, iter, metric; base=nothing, dep=nothing)
    logger[:nrounds] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    isnothing(base) || push!(logger[:base_metrics], base)
    isnothing(dep) || push!(logger[:ctrl_dependence], dep)
    if iter == 0
        logger[:best_metric] = metric
    else
        tol = logger[:early_stopping_tolerance]
        improved = logger[:maximise] ? (metric > logger[:best_metric] + tol) :
                                       (metric < logger[:best_metric] - tol)
        if improved
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end
