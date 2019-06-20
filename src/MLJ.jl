function MLJ.clean!(model::EvoTreeRegressor)
    warning = ""
    if model.nrounds < 1
        warning *= "Need nrounds ≥ 1. Resetting nrounds=1. "
        model.nrounds = 1
    end
    if model.λ < 0
        warning *= "Need λ ≥ 0. Resetting λ=0. "
        model.λ = 0.0
    end
    if model.γ < 0
        warning *= "Need γ ≥ 0. Resetting γ=0. "
        model.γ = 0.0
    end
    if model.η <= 0
        warning *= "Need η > 0. Resetting η=0.001. "
        model.η = 0.001
    end
    if model.max_depth < 1
        warning *= "Need max_depth ≥ 0. Resetting max_depth=0. "
        model.max_depth = 1
    end
    if model.min_weight < 0
        warning *= "Need min_weight ≥ 0. Resetting min_weight=0. "
        model.min_weight = 0.0
    end
    if model.rowsample < 0
        warning *= "Need rowsample ≥ 0. Resetting rowsample=0. "
        model.rowsample = 0.0
    end
    if model.rowsample > 1
        warning *= "Need rowsample <= 1. Resetting rowsample=1. "
        model.rowsample = 1.0
    end
    if model.colsample < 0
        warning *= "Need colsample ≥ 0. Resetting colsample=0. "
        model.colsample = 0.0
    end
    if model.colsample > 1
        warning *= "Need colsample <= 1. Resetting colsample=1. "
        model.colsample = 1.0
    end
    if model.nbins > 250
        warning *= "Need nbins <= 250. Resetting nbins=250. "
        model.nbins = 250
    end
    return warning
end

function MLJBase.fit(model::EvoTreeRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    fitresult = grow_gbtree(Xmatrix, y, model, verbosity = verbosity)
    cache = (Xmatrix, deepcopy(model))
    report = nothing
    return fitresult, cache, report
end

function MLJBase.update(model::EvoTreeRegressor, verbosity,
                        old_fitresult, old_cache, X, y)

    Xmatrix, old_model = old_cache
    δnrounds = model.nrounds - old_model.nrounds 

    # We only continue computation from where we left off if: (i) The
    # number of iterations has not decreased; and (ii) All other
    # hyperparameters are unchanged, with the exception of the
    # learning rate (to allow for externally controlled adaptive
    # learning rate).

    okay_to_continue =
        δnrounds >= 0 &&
        model.loss == old_model.loss &&
        model.λ == old_model.λ &&
        model.γ == old_model.γ &&
        model.max_depth  == old_model.max_depth &&
        model.min_weight == old_model.min_weight &&
        model.rowsample ==  old_model.rowsample &&
        model.colsample ==  old_model.colsample &&
        model.nbins ==  old_model.nbins &&
        model.α ==  old_model.α &&
        model.metric ==  old_model.metric

    if okay_to_continue
        # Grow_gbtree! gets nrounds from old_fitresult.params, which is
        # indentical with model (and which `update` should not modify
        # permanently). For updating, we temporarily change this value
        # to δnrounds:
        old_nrounds = model.nrounds
        model.nrounds = δnrounds

        fitresult = grow_gbtree!(old_fitresult, Xmatrix, y;
                                 verbosity=verbosity)

        # now restore model:
        model.nrounds = old_nrounds
    else
        fitresult = grow_gbtree(Xmatrix, y, model, verbosity = verbosity)
    end

    cache = (Xmatrix, deepcopy(model))
    report = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::EvoTreeRegressor, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    pred = predict(fitresult, Xmatrix)
    return pred
end

# shared metadata
const EvoTypes = Union{EvoTreeRegressor}
MLJBase.input_is_multivariate(::Type{<:EvoTreeRegressor}) = true
MLJBase.input_scitype_union(::Type{<:EvoTreeRegressor}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:EvoTreeRegressor}) = MLJBase.Continuous

MLJBase.package_name(::Type{<:EvoTypes}) = "EvoTrees"
MLJBase.package_url(::Type{<:EvoTypes}) = "https://github.com/Evovest/EvoTrees.jl"
MLJBase.is_pure_julia(::Type{<:EvoTypes}) = true

MLJBase.load_path(::Type{<:EvoTreeRegressor}) = "EvoTrees.MLJ.EvoTreeRegressor"
