struct CallBack{F,M,V,Y}
    feval::F
    x_bin::Matrix{UInt8}
    p::M
    y::Y
    w::V
    feattypes::Vector{Bool}
end

function CallBack(
    params::EvoTypes{L,T},
    m::Union{EvoTree{L,K,T},EvoTreeGPU{L,K,T}};
    metric,
    deval,
    target_name,
    w_name=nothing,
    offset_name=nothing,
    group_name=nothing
) where {L,K,T}
    feval = metric_dict[metric]
    x_bin = binarize(deval; fnames=m.info[:fnames], edges=m.info[:edges])
    p = zeros(T, K, nrow(deval))
    if L == Softmax
        if eltype(deval[!, target_name]) <: CategoricalValue
            levels = CategoricalArrays.levels(deval[!, target_name])
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(deval[!, target_name]))
        else
            levels = sort(unique(deval[!, target_name]))
            yc = CategoricalVector(deval[!, target_name], levels=levels)
            μ = zeros(T, K)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    else
        y = T.(deval[!, target_name])
    end
    w = isnothing(w_name) ? ones(T, size(y)) : Vector{T}(deval[!, w_name])

    offset = !isnothing(offset_name) ? T.(deval[:, offset_name]) : nothing
    if !isnothing(offset)
        L == Logistic && (offset .= logit.(offset))
        L in [Poisson, Gamma, Tweedie] && (offset .= log.(offset))
        L == Softmax && (offset .= log.(offset))
        L in [GaussianMLE, LogisticMLE] && (offset[:, 2] .= log.(offset[:, 2]))
        offset = T.(offset)
        p .+= offset'
    end

    if params.device == "gpu"
        return CallBack(feval, CuArray(x_bin), CuArray(p), CuArray(y), CuArray(w), m.info[:feattypes])
    else
        return CallBack(feval, x_bin, p, y, w, m.info[:feattypes])
    end
end

function (cb::CallBack)(logger, iter, tree)
    predict!(cb.p, tree, cb.x_bin, cb.feattypes)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return nothing
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