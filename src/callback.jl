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

struct CallBack{B,P,Y,C,D,K}
    feval::Function
    x_bin::B
    p::P
    y::Y
    w::C
    eval::C
    feattypes::D
    metric_kwargs::K
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    deval,
    device::Type{<:Device};
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    group_name=nothing) where {L,K}

    T = Float32
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_names = target_name isa AbstractVector ? Symbol.(target_name) : [Symbol(target_name)]

    x_bin = binarize(deval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = length(Tables.getcolumn(deval, 1))
    p = zeros(T, K, nobs)

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
    metric_kwargs = hasproperty(params, :alpha) ? (alpha=T(params.alpha),) : (;)
    if params.metric == :multiquantile
        alphas_eval = T.(params.alphas)
        device <: GPU && (alphas_eval = V{T}(alphas_eval))
        metric_kwargs = (alphas=alphas_eval,)
    end
    if !isnothing(group_name)
        group_eval = build_group_index(Tables.getcolumn(deval, Symbol(group_name)))
        metric_kwargs = merge(metric_kwargs, (group=group_eval,))
    end
    hasproperty(params, :ndcg_k) && (metric_kwargs = merge(metric_kwargs, (ndcg_k=params.ndcg_k,)))

    offset = !isnothing(offset_name) ? T.(Tables.getcolumn(deval, _offset_name)) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && unconstrain_mle_scale!(offset)
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]), metric_kwargs)
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    x_eval::AbstractMatrix,
    y_eval,
    device::Type{<:Device};
    w_eval=nothing,
    offset_eval=nothing,
    group_eval=nothing) where {L,K}

    T = Float32
    x_bin = binarize(x_eval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    p = zeros(T, K, size(x_eval, 1))

    if L == MLogLoss
        y = eval_levelcode(y_eval, m.info[:target_levels])
    else
        y = T.(y_eval)
    end
    feval = metric_dict[params.metric]
    V = device_array_type(device)
    w = isnothing(w_eval) ? device_ones(device, T, size(x_eval, 1)) : V{T}(w_eval)
    metric_kwargs = hasproperty(params, :alpha) ? (alpha=T(params.alpha),) : (;)
    if params.metric == :multiquantile
        alphas_eval = T.(params.alphas)
        device <: GPU && (alphas_eval = V{T}(alphas_eval))
        metric_kwargs = (alphas=alphas_eval,)
    end
    if !isnothing(group_eval)
        metric_kwargs = merge(metric_kwargs, (group=build_group_index(group_eval),))
    end
    hasproperty(params, :ndcg_k) && (metric_kwargs = merge(metric_kwargs, (ndcg_k=params.ndcg_k,)))

    offset = !isnothing(offset_eval) ? T.(offset_eval) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && unconstrain_mle_scale!(offset)
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]), metric_kwargs)
end

function (cb::CallBack)(logger, iter, tree)
    predict!(cb.p, tree, cb.x_bin, cb.feattypes)
    metric = cb.feval(cb.p, cb.y, cb.w, cb.eval; cb.metric_kwargs...)
    update_logger!(logger, iter, metric)
    return nothing
end

function init_logger(; metric, maximise, early_stopping_rounds, early_stopping_tolerance=0.0)
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
    return logger
end

function update_logger!(logger, iter, metric)
    logger[:nrounds] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
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
