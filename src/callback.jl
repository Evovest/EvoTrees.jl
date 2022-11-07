struct CallBack{F,M,V,Y}
    feval::F
    x::M
    p::M
    y::Y
    w::V
end
function (cb::CallBack)(logger, iter, tree)
    predict!(cb.p, tree, cb.x)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return nothing
end

function CallBack(
    params::EvoTypes{L,T},
    ::Union{EvoTree{L,K,T},EvoTreeGPU{L,K,T}};
    metric,
    x_eval,
    y_eval,
    w_eval = nothing,
    offset_eval = nothing,
) where {L,K,T}
    feval = metric_dict[metric]
    x = convert(Matrix{T}, x_eval)
    p = zeros(T, K, length(y_eval))
    if L == Softmax
        if eltype(y_eval) <: CategoricalValue
            levels = CategoricalArrays.levels(y_eval)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(y_eval))
        else
            levels = sort(unique(y_eval))
            yc = CategoricalVector(y_eval, levels = levels)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    else
        y = convert(Vector{T}, y_eval)
    end
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)

    if !isnothing(offset_eval)
        L == Logistic && (offset_eval .= logit.(offset_eval))
        L in [Poisson, Gamma, Tweedie] && (offset_eval .= log.(offset_eval))
        L == Softmax && (offset_eval .= log.(offset_eval))
        L in [GaussianMLE, LogisticMLE] && (offset_eval[:, 2] .= log.(offset_eval[:, 2]))
        offset_eval = T.(offset_eval)
        p .+= offset_eval'
    end

    if params.device == "gpu"
        return CallBack(feval, CuArray(x), CuArray(p), CuArray(y), CuArray(w))
    else
        return CallBack(feval, x, p, y, w)
    end
end

function init_logger(; T, metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :iter => Int[],
        :metrics => T[],
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