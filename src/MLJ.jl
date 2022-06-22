function MMI.fit(model::EvoTypes, verbosity::Int, A, y)

    if model.device == "gpu"
        fitresult, cache = init_evotree_gpu(model, A.matrix, y)
    else
        fitresult, cache = init_evotree(model, A.matrix, y)
    end
    grow_evotree!(fitresult, cache)
    report = (feature_importances=importance(fitresult, A.names),)
    return fitresult, cache, report
end

function okay_to_continue(new, old)
    new.nrounds - old.nrounds >= 0 &&
        new.loss == old.loss &&
        new.λ == old.λ &&
        new.γ == old.γ &&
        new.max_depth == old.max_depth &&
        new.min_weight == old.min_weight &&
        new.rowsample == old.rowsample &&
        new.colsample == old.colsample &&
        new.nbins == old.nbins &&
        new.α == old.α &&
        new.device == old.device &&
        new.metric == old.metric
end


# Generate names to be used by feature_importances in the report 
MMI.reformat(::EvoTypes, X, y) = ((matrix=MMI.matrix(X), names=[name for name ∈ schema(X).names]), y)
MMI.reformat(::EvoTypes, X) = ((matrix=MMI.matrix(X), names=[name for name ∈ schema(X).names]),)
MMI.reformat(::EvoTypes, X::AbstractMatrix, y) = ((matrix=X, names=["feat_$i" for i = 1:size(X, 2)]), y)
MMI.reformat(::EvoTypes, X::AbstractMatrix) = ((matrix=X, names=["feat_$i" for i = 1:size(X, 2)]),)
MMI.selectrows(::EvoTypes, I, A, y) = ((matrix=view(A.matrix, I, :), names=A.names), view(y, I))
MMI.selectrows(::EvoTypes, I, A) = ((matrix=view(A.matrix, I, :), names=A.names),)

# For EarlyStopping.jl support
MMI.iteration_parameter(::Type{<:EvoTypes}) = :nrounds

function MMI.update(model::EvoTypes, verbosity::Integer, fitresult, cache, A, y)

    if okay_to_continue(model, cache.params)
        grow_evotree!(fitresult, cache)
    else
        fitresult, cache = init_evotree(model, A.matrix, y)
        grow_evotree!(fitresult, cache)
    end

    report = (feature_importances=importance(fitresult, A.names),)

    return fitresult, cache, report
end

function predict(::EvoTreeRegressor, fitresult, A)
    pred = vec(predict(fitresult, A.matrix))
    return pred
end

function predict(::EvoTreeClassifier, fitresult, A)
    pred = predict(fitresult, A.matrix)
    return MMI.UnivariateFinite(fitresult.levels, pred, pool=missing)
end

function predict(::EvoTreeCount, fitresult, A)
    λ = vec(predict(fitresult, A.matrix))
    return [Distributions.Poisson(λᵢ) for λᵢ ∈ λ]
end

function predict(::EvoTreeGaussian, fitresult, A)
    pred = predict(fitresult, A.matrix)
    return [Distributions.Normal(pred[i, 1], pred[i, 2]) for i = 1:size(pred, 1)]
end

# Metadata
const EvoTreeRegressor_desc = "Regression models with various underlying methods: least square, quantile, logistic."
const EvoTreeClassifier_desc = "Multi-classification with softmax and cross-entropy loss."
const EvoTreeCount_desc = "Poisson regression fitting λ with max likelihood."
const EvoTreeGaussian_desc = "Gaussian maximum likelihood of μ and σ."

MMI.metadata_pkg.((EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian),
    name="EvoTrees",
    uuid="f6006082-12f8-11e9-0c9c-0d5d367ab1e5",
    url="https://github.com/Evovest/EvoTrees.jl",
    julia=true,
    license="Apache",
    is_wrapper=false)

MMI.metadata_model(EvoTreeRegressor,
    input=Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    target=AbstractVector{<:MMI.Continuous},
    weights=false,
    path="EvoTrees.EvoTreeRegressor",
    descr=EvoTreeRegressor_desc)

MMI.metadata_model(EvoTreeClassifier,
    input=Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    target=AbstractVector{<:MMI.Finite},
    weights=false,
    path="EvoTrees.EvoTreeClassifier",
    descr=EvoTreeClassifier_desc)

MMI.metadata_model(EvoTreeCount,
    input=Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    target=AbstractVector{<:MMI.Count},
    weights=false,
    path="EvoTrees.EvoTreeCount",
    descr=EvoTreeCount_desc)

MMI.metadata_model(EvoTreeGaussian,
    input=Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    target=AbstractVector{<:MMI.Continuous},
    weights=false,
    path="EvoTrees.EvoTreeGaussian",
    descr=EvoTreeGaussian_desc)

"""
$(MMI.doc_header(EvoTreeRegressor))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `loss`:               One of `:linear`, `:logistic`, `:quantile`, `:L1`
- `nrounds=10`:         Max number of rounds
- `rng=Random.GLOBAL_RNG`: random number generator or seed

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

# Fitted parameters

The fields of `fitted_params(mach)` are:
- `gbtree`: The GBTree object returned by EvoTrees.jl fitting algorithm

# Report

The fields of `report(mach)` are:
- ...

# Examples

```
using MLJ
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeClassifier(max_depth=5, num_bins=32)
X, y = @load_crab
mach = machine(model, X, y) |> fit!
```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeRegressor


"""
$(MMI.doc_header(EvoTreeClassifier))

# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Finite`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `nrounds=10`:         Max number of rounds
- `rng=Random.GLOBAL_RNG`: random number generator or seed

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

# Fitted parameters

The fields of `fitted_params(mach)` are:
- `gbtree`: The GBTree object returned by EvoTrees.jl fitting algorithm

# Report

The fields of `report(mach)` are:
- ...

# Examples

```
using MLJ
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier(max_depth=5, num_bins=32)
X, y = @load_iris
mach = machine(model, X, y) |> fit!
```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeClassifier

"""
$(MMI.doc_header(EvoTreeCount))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Count`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `nrounds=10`:         Max number of rounds
- `rng=Random.GLOBAL_RNG`: random number generator or seed

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

# Fitted parameters

The fields of `fitted_params(mach)` are:
- `gbtree`: The GBTree object returned by EvoTrees.jl fitting algorithm

# Report

The fields of `report(mach)` are:
- ...

# Examples

```
using MLJ
EvoTreeCount = @load EvoTreeCount pkg=EvoTrees
model = EvoTreeCount(max_depth=5, num_bins=32)
X, y = @load_crab
mach = machine(model, X, y) |> fit!
```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeCount

"""
$(MMI.doc_header(EvoTreeGaussian))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `nrounds=10`:         Max number of rounds
- `rng=Random.GLOBAL_RNG`: random number generator or seed

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

# Fitted parameters

The fields of `fitted_params(mach)` are:
- `gbtree`: The GBTree object returned by EvoTrees.jl fitting algorithm

# Report

The fields of `report(mach)` are:
- ...

# Examples

```
using MLJ
EvoTreeGaussian = @load EvoTreeGaussian pkg=EvoTrees
model = EvoTreeGaussian(max_depth=5, num_bins=32)
X, y = @load_crab
mach = machine(model, X, y) |> fit!
```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeGaussian

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
