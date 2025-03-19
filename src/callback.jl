struct CallBack{B,P,Y,C,D}
    feval::Function
    x_bin::B
    p::P
    y::Y
    w::C
    eval::C
    feattypes::D
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    deval,
    device::Type{<:Device};
    target_name,
    weight_name=nothing,
    offset_name=nothing) where {L,K}

    T = Float32
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_name = Symbol(target_name)

    feval = metric_dict[params.metric]
    x_bin = binarize(deval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs = length(Tables.getcolumn(deval, 1))
    p = zeros(T, K, nobs)

    y_eval = Tables.getcolumn(deval, _target_name)

    if L == MLogLoss
        if eltype(y_eval) <: CategoricalValue
            levels = CategoricalArrays.levels(y_eval)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(y_eval))
        else
            levels = sort(unique(y_eval))
            yc = CategoricalVector(y_eval, levels=levels)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    else
        y = T.(y_eval)
    end
    V = device_array_type(device)
    w = isnothing(weight_name) ? device_ones(device, T, length(y)) : V{T}(Tables.getcolumn(deval, _weight_name))

    offset = !isnothing(offset_name) ? T.(Tables.getcolumn(deval, _offset_name)) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MultiClassRegression && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && (offset[:, 2] .= log.(offset[:, 2]))
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]))
end

function CallBack(
    params::EvoTypes,
    m::EvoTree{L,K},
    x_eval::AbstractMatrix,
    y_eval,
    device::Type{<:Device};
    w_eval=nothing,
    offset_eval=nothing) where {L,K}

    T = Float32
    feval = metric_dict[params.metric]
    x_bin = binarize(x_eval; feature_names=m.info[:feature_names], edges=m.info[:edges])
    p = zeros(T, K, size(x_eval, 1))

    if L == MLogLoss
        if eltype(y_eval) <: CategoricalValue
            levels = CategoricalArrays.levels(y_eval)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(y_eval))
        else
            levels = sort(unique(y_eval))
            yc = CategoricalVector(y_eval, levels=levels)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    else
        y = T.(y_eval)
    end
    V = device_array_type(device)
    w = isnothing(w_eval) ? device_ones(device, T, length(y)) : V{T}(w_eval)

    offset = !isnothing(offset_eval) ? T.(offset_eval) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && (offset[:, 2] .= log.(offset[:, 2]))
        offset = T.(offset)
        p .+= offset'
    end

    return CallBack(feval, convert(V, x_bin), convert(V, p), convert(V, y), w, similar(w), convert(V, m.info[:feattypes]))
end

function (cb::CallBack)(logger, iter, tree)
    predict!(cb.p, tree, cb.x_bin, cb.feattypes)
    metric = cb.feval(cb.p, cb.y, cb.w, cb.eval)
    update_logger!(logger, iter, metric)
    return nothing
end

function init_logger(; metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :iter => Int[],
        :metrics => Float32[],
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
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end
