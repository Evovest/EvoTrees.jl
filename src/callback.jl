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
    ::EvoTypes{L},
    m::EvoTree{L,K},
    deval,
    device::Type{<:Device};
    target_name,
    w_name=nothing,
    offset_name=nothing,
    metric) where {L,K}

    T = Float32
    _w_name = isnothing(w_name) ? Symbol("") : Symbol(w_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_name = Symbol(target_name)

    feval = metric_dict[metric]
    x_bin = binarize(deval; fnames=m.info[:fnames], edges=m.info[:edges])
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
    w = isnothing(w_name) ? ones(T, size(y)) : Vector{T}(Tables.getcolumn(deval, _w_name))

    offset = !isnothing(offset_name) ? T.(Tables.getcolumn(deval, _offset_name)) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MultiClassRegression && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && (offset[:, 2] .= log.(offset[:, 2]))
        offset = T.(offset)
        p .+= offset'
    end

    if device <: GPU
        return CallBack(feval, CuArray(x_bin), CuArray(p), CuArray(y), CuArray(w), CuArray(similar(w)), CuArray(m.info[:feattypes]))
    else
        return CallBack(feval, x_bin, p, y, w, similar(w), m.info[:feattypes])
    end
end

function CallBack(
    ::EvoTypes{L},
    m::EvoTree{L,K},
    x_eval::AbstractMatrix,
    y_eval,
    device::Type{<:Device};
    w_eval=nothing,
    offset_eval=nothing,
    metric) where {L,K}

    T = Float32
    feval = metric_dict[metric]
    x_bin = binarize(x_eval; fnames=m.info[:fnames], edges=m.info[:edges])
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
    w = isnothing(w_eval) ? ones(T, size(y)) : Vector{T}(w_eval)

    offset = !isnothing(offset_eval) ? T.(offset_eval) : nothing
    if !isnothing(offset)
        L == LogLoss && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == MLogLoss && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && (offset[:, 2] .= log.(offset[:, 2]))
        offset = T.(offset)
        p .+= offset'
    end

    if device <: GPU
        return CallBack(feval, CuArray(x_bin), CuArray(p), CuArray(y), CuArray(w), CuArray(similar(w)), CuArray(m.info[:feattypes]))
    else
        return CallBack(feval, x_bin, p, y, w, similar(w), m.info[:feattypes])
    end
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