
function MLJModelInterface.fit(model::EvoTypes, verbosity::Int, X, y)
    Xmatrix = MLJModelInterface.matrix(X)
    fitresult, cache = init_evotree(model, Xmatrix, y, verbosity = verbosity)
    grow_evotree!(fitresult, cache, verbosity = verbosity)
    report = nothing
    return fitresult, cache, report
end

function okay_to_continue(new, old)
    new.nrounds - old.nrounds >= 0 &&
    new.loss == old.loss &&
    new.λ == old.λ &&
    new.γ == old.γ &&
    new.max_depth  == old.max_depth &&
    new.min_weight == old.min_weight &&
    new.rowsample ==  old.rowsample &&
    new.colsample ==  old.colsample &&
    new.nbins ==  old.nbins &&
    new.α ==  old.α &&
    new.metric ==  old.metric
end

function MLJModelInterface.update(model::EvoTypes, verbosity::Integer, fitresult, cache, X, y)

    if okay_to_continue(model, cache.params)
        grow_evotree!(fitresult, cache, verbosity = verbosity)
    else
        Xmatrix = MLJModelInterface.matrix(X)
        fitresult, cache = init_evotree(model, Xmatrix, y, verbosity = verbosity)
        grow_evotree!(fitresult, cache, verbosity = verbosity)
    end
    report = nothing
    return fitresult, cache, report
end

function predict(model::EvoTreeRegressor, fitresult, Xnew)
    Xnew = MLJModelInterface.matrix(Xnew)
    pred = predict(fitresult, Xnew)
    return pred
end

function predict(model::EvoTreeClassifier, fitresult, Xnew)
    Xnew = MLJModelInterface.matrix(Xnew)
    pred = predict(fitresult, Xnew)
    return MLJModelInterface.UnivariateFinite(fitresult.levels, pred)
end

function predict(model::EvoTreeCount, fitresult, Xnew)
    Xnew = MLJModelInterface.matrix(Xnew)
    λ = predict(fitresult, Xnew)
    return [Distributions.Poisson(λᵢ) for λᵢ ∈ λ]
end

function predict(model::EvoTreeGaussian, fitresult, Xnew)
    Xnew = MLJModelInterface.matrix(Xnew)
    pred = predict(fitresult, Xnew)
    return [Distributions.Normal(pred[i,1], sqrt(pred[i,2])) for i in 1:size(pred,1)]
end

# Metadata
const EvoTreeRegressor_desc = "Regression models with various underlying methods: least square, quantile, logistic."
const EvoTreeClassifier_desc = "Multi-classification with softmax and cross-entropy loss."
const EvoTreeCount_desc = "Poisson regression fitting λ with max likelihood."
const EvoTreeGaussian_desc = "Gaussian maximum likelihood of μ and σ²."

MLJModelInterface.metadata_pkg.((EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian),
    name="EvoTrees",
    uuid="f6006082-12f8-11e9-0c9c-0d5d367ab1e5",
    url="https://github.com/Evovest/EvoTrees.jl",
    julia=true,
    license="Apache",
    is_wrapper=false)

MLJModelInterface.metadata_model(EvoTreeRegressor,
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{<:MLJModelInterface.Continuous},
    weights=false,
    path="EvoTrees.EvoTreeRegressor",
    descr=EvoTreeRegressor_desc)

MLJModelInterface.metadata_model(EvoTreeClassifier,
        input=MLJModelInterface.Table(MLJModelInterface.Continuous),
        target=AbstractVector{<:MLJModelInterface.Finite},
        weights=false,
        path="EvoTrees.EvoTreeClassifier",
        descr=EvoTreeClassifier_desc)

MLJModelInterface.metadata_model(EvoTreeCount,
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{<:MLJModelInterface.Count},
    weights=false,
    path="EvoTrees.EvoTreeCount",
    descr=EvoTreeCount_desc)

MLJModelInterface.metadata_model(EvoTreeGaussian,
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{<:MLJModelInterface.Continuous},
    weights=false,
    path="EvoTrees.EvoTreeGaussian",
    descr=EvoTreeGaussian_desc)

# shared metadata
# MLJModelInterface.package_name(::Type{<:EvoTypes}) = "EvoTrees"
# MLJModelInterface.package_uuid(::Type{<:EvoTypes}) = "f6006082-12f8-11e9-0c9c-0d5d367ab1e5"
# MLJModelInterface.package_url(::Type{<:EvoTypes}) = "https://github.com/Evovest/EvoTrees.jl"
# MLJModelInterface.is_pure_julia(::Type{<:EvoTypes}) = true
#
# MLJModelInterface.load_path(::Type{<:EvoTreeRegressor}) = "EvoTrees.EvoTreeRegressor"
# MLJModelInterface.input_scitype(::Type{<:EvoTreeRegressor}) = MLJModelInterface.Table(MLJModelInterface.Continuous)
# MLJModelInterface.target_scitype(::Type{<:EvoTreeRegressor}) = AbstractVector{<:MLJModelInterface.Continuous}
#
# MLJModelInterface.load_path(::Type{<:EvoTreeCount}) = "EvoTrees.EvoTreeCount"
# MLJModelInterface.input_scitype(::Type{<:EvoTreeCount}) = MLJModelInterface.Table(MLJModelInterface.Continuous)
# MLJModelInterface.target_scitype(::Type{<:EvoTreeCount}) = AbstractVector{<:MLJModelInterface.Count}
#
# MLJModelInterface.load_path(::Type{<:EvoTreeClassifier}) = "EvoTrees.EvoTreeClassifier"
# MLJModelInterface.input_scitype(::Type{<:EvoTreeClassifier}) = MLJModelInterface.Table(MLJModelInterface.Continuous)
# MLJModelInterface.target_scitype(::Type{<:EvoTreeClassifier}) = AbstractVector{<:MLJModelInterface.Finite}
#
# MLJModelInterface.load_path(::Type{<:EvoTreeGaussian}) = "EvoTrees.EvoTreeGaussian"
# MLJModelInterface.input_scitype(::Type{<:EvoTreeGaussian}) = MLJModelInterface.Table(MLJModelInterface.Continuous)
# MLJModelInterface.target_scitype(::Type{<:EvoTreeGaussian}) = AbstractVector{<:MLJModelInterface.Continuous}

# function MLJ.clean!(model::EvoTreeRegressor)
#     warning = ""
#     if model.nrounds < 1
#         warning *= "Need nrounds ≥ 1. Resetting nrounds=1. "
#         model.nrounds = 1
#     end
#     if model.λ < 0
#         warning *= "Need λ ≥ 0. Resetting λ=0. "
#         model.λ = 0.0
#     end
#     if model.γ < 0
#         warning *= "Need γ ≥ 0. Resetting γ=0. "
#         model.γ = 0.0
#     end
#     if model.η <= 0
#         warning *= "Need η > 0. Resetting η=0.001. "
#         model.η = 0.001
#     end
#     if model.max_depth < 1
#         warning *= "Need max_depth ≥ 0. Resetting max_depth=0. "
#         model.max_depth = 1
#     end
#     if model.min_weight < 0
#         warning *= "Need min_weight ≥ 0. Resetting min_weight=0. "
#         model.min_weight = 0.0
#     end
#     if model.rowsample < 0
#         warning *= "Need rowsample ≥ 0. Resetting rowsample=0. "
#         model.rowsample = 0.0
#     end
#     if model.rowsample > 1
#         warning *= "Need rowsample <= 1. Resetting rowsample=1. "
#         model.rowsample = 1.0
#     end
#     if model.colsample < 0
#         warning *= "Need colsample ≥ 0. Resetting colsample=0. "
#         model.colsample = 0.0
#     end
#     if model.colsample > 1
#         warning *= "Need colsample <= 1. Resetting colsample=1. "
#         model.colsample = 1.0
#     end
#     if model.nbins > 250
#         warning *= "Need nbins <= 250. Resetting nbins=250. "
#         model.nbins = 250
#     end
#     return warning
# end
