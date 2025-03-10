function MMI.fit(model::EvoTypes, verbosity::Int, A, y, w=nothing)

  A = isa(A, AbstractMatrix) ? Tables.columntable(Tables.table(A)) : Tables.columntable(A)
  nobs = Tables.DataAPI.nrow(A)
  feature_names = collect(Tables.schema(A).names)

  T = Float32
  device = model.device == :gpu ? GPU : CPU
  V = device_array_type(device)
  w = isnothing(w) ? device_ones(device, T, nobs) : V{T}(w)
  fitresult, cache = init_core(model, device, A, feature_names, y, w, nothing)

  while fitresult.info[:nrounds] < model.nrounds
    grow_evotree!(fitresult, cache, model)
  end
  report = (features=cache.feature_names,)
  return fitresult, cache, report
end

function okay_to_continue(model, fitresult, cache)
  check = model.nrounds - fitresult.info[:nrounds] >= 0
  return check
end

# For EarlyStopping.jl support
MMI.iteration_parameter(::Type{<:EvoTypes}) = :nrounds

function MMI.update(
  model::EvoTypes,
  verbosity::Integer,
  fitresult,
  cache,
  A,
  y,
  w=nothing,
)
  if okay_to_continue(model, fitresult, cache)
    while fitresult.info[:nrounds] < model.nrounds
      grow_evotree!(fitresult, cache, model)
    end
    report = (features=cache.feature_names,)
  else
    fitresult, cache, report = fit(model, verbosity, A, y, w)
  end
  return fitresult, cache, report
end

function predict(::EvoTreeRegressor, fitresult::EvoTree, A)
  pred = predict(fitresult, A)
  return pred
end

function predict(::EvoTreeClassifier, fitresult::EvoTree, A)
  pred = predict(fitresult, A)
  return MMI.UnivariateFinite(fitresult.info[:target_levels], pred, pool=missing, ordered=fitresult.info[:target_isordered])
end

function predict(::EvoTreeCount, fitresult::EvoTree, A)
  λs = predict(fitresult, A)
  return [Distributions.Poisson(λ) for λ ∈ λs]
end

function predict(::EvoTreeGaussian, fitresult::EvoTree, A)
  pred = predict(fitresult, A)
  return [Distributions.Normal(pred[i, 1], pred[i, 2]) for i in axes(pred, 1)]
end

function predict(::EvoTreeMLE, fitresult::EvoTree{L,K}, A) where {L<:GaussianMLE,K}
  pred = predict(fitresult, A)
  return [Distributions.Normal(pred[i, 1], pred[i, 2]) for i in axes(pred, 1)]
end

function predict(::EvoTreeMLE, fitresult::EvoTree{L,K}, A) where {L<:LogisticMLE,K}
  pred = predict(fitresult, A)
  return [Distributions.Logistic(pred[i, 1], pred[i, 2]) for i in axes(pred, 1)]
end

function feature_importances(m::EvoTypes, fitresult, report)
  fi_pairs = importance(fitresult, feature_names=report[:features])
  return fi_pairs
end


# Metadata

MMI.metadata_pkg.(
  (EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian, EvoTreeMLE),
  name="EvoTrees",
  uuid="f6006082-12f8-11e9-0c9c-0d5d367ab1e5",
  url="https://github.com/Evovest/EvoTrees.jl",
  julia=true,
  license="Apache",
  is_wrapper=false,
)

MMI.metadata_model(
  EvoTreeRegressor,
  input_scitype=Union{
    MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    AbstractMatrix{MMI.Continuous},
  },
  target_scitype=AbstractVector{<:MMI.Continuous},
  supports_weights=true,
  reports_feature_importances=true,
  path="EvoTrees.EvoTreeRegressor",
)

MMI.metadata_model(
  EvoTreeClassifier,
  input_scitype=Union{
    MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    AbstractMatrix{MMI.Continuous},
  },
  target_scitype=AbstractVector{<:MMI.Finite},
  supports_weights=true,
  reports_feature_importances=true,
  path="EvoTrees.EvoTreeClassifier",
)

MMI.metadata_model(
  EvoTreeCount,
  input_scitype=Union{
    MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    AbstractMatrix{MMI.Continuous},
  },
  target_scitype=AbstractVector{<:MMI.Count},
  supports_weights=true,
  reports_feature_importances=true,
  path="EvoTrees.EvoTreeCount",
)

MMI.metadata_model(
  EvoTreeGaussian,
  input_scitype=Union{
    MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    AbstractMatrix{MMI.Continuous},
  },
  target_scitype=AbstractVector{<:MMI.Continuous},
  supports_weights=true,
  reports_feature_importances=true,
  path="EvoTrees.EvoTreeGaussian",
)

MMI.metadata_model(
  EvoTreeMLE,
  input_scitype=Union{
    MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
    AbstractMatrix{MMI.Continuous},
  },
  target_scitype=AbstractVector{<:MMI.Continuous},
  supports_weights=true,
  reports_feature_importances=true,
  path="EvoTrees.EvoTreeMLE",
)

"""
  EvoTreeRegressor(;kwargs...)

A model type for constructing a EvoTreeRegressor, based on [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `loss=:mse`:         Loss to be be minimized during training. One of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:gamma`
  - `:tweedie`
  - `:quantile`
  - `:cred_var`: **experimental** credibility-based gains, derived from ratio of spread to process variance.
  - `:cred_std`: **experimental** credibility-based gains, derived from ratio of spread to process std deviation.
- `metric`:     The evaluation metric used to track evaluation data and serves as a basis for early stopping. Supported metrics are: 
  - `:mse`:     Mean-squared error. Adapted for general regression models.
  - `:rmse`:    Root-mean-squared error. Adapted for general regression models.
  - `:mae`:     Mean absolute error. Adapted for general regression models.
  - `:logloss`: Adapted for `:logistic` regression models.
  - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
  - `:gamma`:   Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
  - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
  - `:quantile`: The corresponds to an assymetric absolute error, where residuals are penalized according to alpha / (1-alpha) according to their sign.
  - `:gini`: The normalized Gini between pred and target
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `nrounds=100`:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked. Must be >= 1.
- `eta=0.1`:              Learning rate. Each tree raw predictions are scaled by `eta` prior to be added to the stack of predictions. Must be > 0.
  A lower `eta` results in slower learning, requiring a higher `nrounds` but typically improves model performance.   
- `L2::T=0.0`:            L2 regularization factor on aggregate gain. Must be >= 0. Higher L2 can result in a more robust model.
- `lambda::T=0.0`:        L2 regularization factor on individual gain. Must be >= 0. Higher lambda can result in a more robust model.
- `gamma::T=0.0`:         Minimum gain improvement needed to perform a node split. Higher gamma can result in a more robust model. Must be >= 0.
- `alpha::T=0.5`:         Loss specific parameter in the [0, 1] range:
                            - `:quantile`: target quantile for the regression.
- `max_depth=6`:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.
  A complete tree of depth N contains `2^(N - 1)` terminal leaves and `2^(N - 1) - 1` split nodes.
  Compute cost is proportional to `2^max_depth`. Typical optimal values are in the 3 to 9 range.
- `min_weight=1.0`:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the `weights` vector. Must be > 0.
- `rowsample=1.0`:        Proportion of rows that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `colsample=1.0`:        Proportion of columns / features that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `nbins=64`:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins. Should be between 2 and 255.
- `monotone_constraints=Dict{Int, Int}()`: Specify monotonic constraints using a dict where the key is the feature index and the value the applicable constraint (-1=decreasing, 0=none, 1=increasing). 
  Only `:linear`, `:logistic`, `:gamma` and `tweedie` losses are supported at the moment.
- `tree_type=:binary`    Tree structure to be used. One of:
  - `:binary`:       Each node of a tree is grown independently. Tree are built depthwise until max depth is reach or if min weight or gain (see `gamma`) stops further node splits.  
  - `:oblivious`:    A common splitting condition is imposed to all nodes of a given depth. 
- `rng=123`:              Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`: Hardware device to use for computations. Can be either `:cpu` or `gpu`.

# Internal API

Do `config = EvoTreeRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeRegressor(loss=...).

## Training model

A model is built using [`fit_evotree`](@ref):

```julia
model = fit_evotree(config; x_train, y_train, kwargs...)
```

## Inference

Predictions are obtained using [`predict`](@ref) which returns a `Vector` of length `nobs`:

```julia
EvoTrees.predict(model, X)
```

Alternatively, models act as a functor, returning predictions when called as a function with features as argument:

```julia
model(X)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
```

Do `model = EvoTreeRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoTreeRegressor(loss=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with
    `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

## Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `GBTree` object returned by EvoTrees.jl fitting algorithm.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

```
# Internal API
using EvoTrees
config = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
```

```
# MLJ Interface
using MLJ
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor(max_depth=5, nbins=32, nrounds=100)
X, y = @load_boston
mach = machine(model, X, y) |> fit!
preds = predict(mach, X)
```
"""
EvoTreeRegressor


"""
  EvoTreeClassifier(;kwargs...)

A model type for constructing a EvoTreeClassifier, based on [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl), and implementing both an internal API and the MLJ model interface.
EvoTreeClassifier is used to perform multi-class classification, using cross-entropy loss.

# Hyper-parameters

- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `nrounds=100`:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked. Must be >= 1.
- `eta=0.1`:              Learning rate. Each tree raw predictions are scaled by `eta` prior to be added to the stack of predictions. Must be > 0.
  A lower `eta` results in slower learning, requiring a higher `nrounds` but typically improves model performance.  
- `L2::T=0.0`:            L2 regularization factor on aggregate gain. Must be >= 0. Higher L2 can result in a more robust model.
- `lambda::T=0.0`:        L2 regularization factor on individual gain. Must be >= 0. Higher lambda can result in a more robust model.
- `gamma::T=0.0`:         Minimum gain improvement needed to perform a node split. Higher gamma can result in a more robust model. Must be >= 0.
- `max_depth=6`:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.
  A complete tree of depth N contains `2^(N - 1)` terminal leaves and `2^(N - 1) - 1` split nodes.
  Compute cost is proportional to `2^max_depth`. Typical optimal values are in the 3 to 9 range.
- `min_weight=1.0`:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the `weights` vector. Must be > 0.
- `rowsample=1.0`:        Proportion of rows that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `colsample=1.0`:        Proportion of columns / features that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `nbins=64`:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins. Should be between 2 and 255.
- `tree_type=:binary`    Tree structure to be used. One of:
  - `:binary`:       Each node of a tree is grown independently. Tree are built depthwise until max depth is reach or if min weight or gain (see `gamma`) stops further node splits.  
  - `:oblivious`:    A common splitting condition is imposed to all nodes of a given depth. 
- `rng=123`:              Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`: Hardware device to use for computations. Can be either `:cpu` or `:gpu`.

# Internal API

Do `config = EvoTreeClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeClassifier(max_depth=...).

## Training model

A model is built using [`fit_evotree`](@ref):

```julia
model = fit_evotree(config; x_train, y_train, kwargs...)
```

## Inference

Predictions are obtained using [`predict`](@ref) which returns a `Matrix` of size `[nobs, K]` where `K` is the number of classes:

```julia
EvoTrees.predict(model, X)
```

Alternatively, models act as a functor, returning predictions when called as a function with features as argument:

```julia
model(X)
```

# MLJ

From MLJ, the type can be imported using:

```julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
```

Do `model = EvoTreeClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoTreeClassifier(loss=...)`.

## Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Multiclas` or `<:OrderedFactor`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

## Operations

- `predict(mach, Xnew)`: return predictions of the target given features `Xnew` having the same scitype as `X` above.
  Predictions are probabilistic.

- `predict_mode(mach, Xnew)`: returns the mode of each of the prediction above.

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `GBTree` object returned by EvoTrees.jl fitting algorithm.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

```
# Internal API
using EvoTrees
config = EvoTreeClassifier(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(1:3, nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
```

```
# MLJ Interface
using MLJ
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier(max_depth=5, nbins=32, nrounds=100)
X, y = @load_iris
mach = machine(model, X, y) |> fit!
preds = predict(mach, X)
preds = predict_mode(mach, X)
```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeClassifier

"""
  EvoTreeCount(;kwargs...)

A model type for constructing a EvoTreeCount, based on [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl), and implementing both an internal API the MLJ model interface.
EvoTreeCount is used to perform Poisson probabilistic regression on count target.

# Hyper-parameters

- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `nrounds=100`:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked. Must be >= 1.
- `eta=0.1`:              Learning rate. Each tree raw predictions are scaled by `eta` prior to be added to the stack of predictions. Must be > 0.
  A lower `eta` results in slower learning, requiring a higher `nrounds` but typically improves model performance.  
- `L2::T=0.0`:            L2 regularization factor on aggregate gain. Must be >= 0. Higher L2 can result in a more robust model.
- `lambda::T=0.0`:        L2 regularization factor on individual gain. Must be >= 0. Higher lambda can result in a more robust model.
- `gamma::T=0.0`:         Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model.
- `max_depth=6`:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.
  A complete tree of depth N contains `2^(N - 1)` terminal leaves and `2^(N - 1) - 1` split nodes.
  Compute cost is proportional to 2^max_depth. Typical optimal values are in the 3 to 9 range.
- `min_weight=1.0`:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the `weights` vector. Must be > 0.
- `rowsample=1.0`:        Proportion of rows that are sampled at each iteration to build the tree. Should be `]0, 1]`.
- `colsample=1.0`:        Proportion of columns / features that are sampled at each iteration to build the tree. Should be `]0, 1]`.
- `nbins=64`:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins. Should be between 2 and 255.
- `monotone_constraints=Dict{Int, Int}()`: Specify monotonic constraints using a dict where the key is the feature index and the value the applicable constraint (-1=decreasing, 0=none, 1=increasing).
- `tree_type=:binary`    Tree structure to be used. One of:
  - `:binary`:       Each node of a tree is grown independently. Tree are built depthwise until max depth is reach or if min weight or gain (see `gamma`) stops further node splits.  
  - `:oblivious`:    A common splitting condition is imposed to all nodes of a given depth. 
- `rng=123`:              Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`: Hardware device to use for computations. Can be either `:cpu` or `:gpu`.

# Internal API

Do `config = EvoTreeCount()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeCount(max_depth=...).

## Training model

A model is built using [`fit_evotree`](@ref):

```julia
model = fit_evotree(config; x_train, y_train, kwargs...)
```

## Inference

Predictions are obtained using [`predict`](@ref) which returns a `Vector` of length `nobs`:

```julia
EvoTrees.predict(model, X)
```

Alternatively, models act as a functor, returning predictions when called as a function with features as argument:

```julia
model(X)
```

# MLJ

From MLJ, the type can be imported using:

```julia
EvoTreeCount = @load EvoTreeCount pkg=EvoTrees
```

Do `model = EvoTreeCount()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoTreeCount(loss=...)`.

## Training data

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

# Operations

- `predict(mach, Xnew)`: returns a vector of Poisson distributions given features `Xnew`
  having the same scitype as `X` above. Predictions are probabilistic.

Specific metrics can also be predicted using:

  - `predict_mean(mach, Xnew)`
  - `predict_mode(mach, Xnew)`
  - `predict_median(mach, Xnew)`

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `GBTree` object returned by EvoTrees.jl fitting algorithm.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

```
# Internal API
using EvoTrees
config = EvoTreeCount(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(0:2, nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
```

```
using MLJ
EvoTreeCount = @load EvoTreeCount pkg=EvoTrees
model = EvoTreeCount(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
X, y = randn(nobs, nfeats), rand(0:2, nobs)
mach = machine(model, X, y) |> fit!
preds = predict(mach, X)
preds = predict_mean(mach, X)
preds = predict_mode(mach, X)
preds = predict_median(mach, X)

```

See also
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).
"""
EvoTreeCount

"""
  EvoTreeGaussian(;kwargs...)

A model type for constructing a EvoTreeGaussian, based on [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl), and implementing both an internal API the MLJ model interface.
EvoTreeGaussian is used to perform Gaussian probabilistic regression, fitting μ and σ parameters to maximize likelihood.

# Hyper-parameters

- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `nrounds=100`:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked. Must be >= 1.
- `eta=0.1`:              Learning rate. Each tree raw predictions are scaled by `eta` prior to be added to the stack of predictions. Must be > 0.
  A lower `eta` results in slower learning, requiring a higher `nrounds` but typically improves model performance.  
- `L2::T=0.0`:            L2 regularization factor on aggregate gain. Must be >= 0. Higher L2 can result in a more robust model.
- `lambda::T=0.0`:        L2 regularization factor on individual gain. Must be >= 0. Higher lambda can result in a more robust model.
- `gamma::T=0.0`:         Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model. Must be >= 0.
- `max_depth=6`:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.
  A complete tree of depth N contains `2^(N - 1)` terminal leaves and `2^(N - 1) - 1` split nodes.
  Compute cost is proportional to 2^max_depth. Typical optimal values are in the 3 to 9 range.
- `min_weight=8.0`:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the `weights` vector. Must be > 0.
- `rowsample=1.0`:        Proportion of rows that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `colsample=1.0`:        Proportion of columns / features that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `nbins=64`:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins. Should be between 2 and 255.
- `monotone_constraints=Dict{Int, Int}()`: Specify monotonic constraints using a dict where the key is the feature index and the value the applicable constraint (-1=decreasing, 0=none, 1=increasing). 
  !Experimental feature: note that for Gaussian regression, constraints may not be enforce systematically.
- `tree_type=:binary`    Tree structure to be used. One of:
  - `:binary`:       Each node of a tree is grown independently. Tree are built depthwise until max depth is reach or if min weight or gain (see `gamma`) stops further node splits.  
  - `:oblivious`:    A common splitting condition is imposed to all nodes of a given depth. 
- `rng=123`:              Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`: Hardware device to use for computations. Can be either `:cpu` or `gpu`.

# Internal API

Do `config = EvoTreeGaussian()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeGaussian(max_depth=...).

## Training model

A model is built using [`fit_evotree`](@ref):

```julia
model = fit_evotree(config; x_train, y_train, kwargs...)
```

## Inference

Predictions are obtained using [`predict`](@ref) which returns a `Matrix` of size `[nobs, 2]` where the second dimensions refer to `μ` and `σ` respectively:

```julia
EvoTrees.predict(model, X)
```

Alternatively, models act as a functor, returning predictions when called as a function with features as argument:

```julia
model(X)
```

# MLJ

From MLJ, the type can be imported using:

```julia
EvoTreeGaussian = @load EvoTreeGaussian pkg=EvoTrees
```

Do `model = EvoTreeGaussian()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoTreeGaussian(loss=...)`.

## Training data

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

## Operations

- `predict(mach, Xnew)`: returns a vector of Gaussian distributions given features `Xnew` having the same scitype as `X` above.
Predictions are probabilistic.

Specific metrics can also be predicted using:

  - `predict_mean(mach, Xnew)`
  - `predict_mode(mach, Xnew)`
  - `predict_median(mach, Xnew)`

## Fitted parameters

The fields of `fitted_params(mach)` are:

  - `:fitresult`: The `GBTree` object returned by EvoTrees.jl fitting algorithm.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

```
# Internal API
using EvoTrees
params = EvoTreeGaussian(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(nobs)
model = fit_evotree(params; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
```

```
# MLJ Interface
using MLJ
EvoTreeGaussian = @load EvoTreeGaussian pkg=EvoTrees
model = EvoTreeGaussian(max_depth=5, nbins=32, nrounds=100)
X, y = @load_boston
mach = machine(model, X, y) |> fit!
preds = predict(mach, X)
preds = predict_mean(mach, X)
preds = predict_mode(mach, X)
preds = predict_median(mach, X)
```
"""
EvoTreeGaussian



"""
  EvoTreeMLE(;kwargs...)

A model type for constructing a EvoTreeMLE, based on [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl), and implementing both an internal API the MLJ model interface.
EvoTreeMLE performs maximum likelihood estimation. Assumed distribution is specified through `loss` kwargs. Both Gaussian and Logistic distributions are supported.

# Hyper-parameters

- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
`loss=:gaussian`:         Loss to be be minimized during training. One of:
  - `:gaussian_mle`
  - `:logistic_mle`
- `nrounds=100`:           Number of rounds. It corresponds to the number of trees that will be sequentially stacked. Must be >= 1.
- `eta=0.1`:              Learning rate. Each tree raw predictions are scaled by `eta` prior to be added to the stack of predictions. Must be > 0.
  A lower `eta` results in slower learning, requiring a higher `nrounds` but typically improves model performance.  
- `L2::T=0.0`:            L2 regularization factor on aggregate gain. Must be >= 0. Higher L2 can result in a more robust model.
- `lambda::T=0.0`:        L2 regularization factor on individual gain. Must be >= 0. Higher lambda can result in a more robust model.
- `gamma::T=0.0`:         Minimum gain imprvement needed to perform a node split. Higher gamma can result in a more robust model. Must be >= 0.
- `max_depth=6`:          Maximum depth of a tree. Must be >= 1. A tree of depth 1 is made of a single prediction leaf.
  A complete tree of depth N contains `2^(N - 1)` terminal leaves and `2^(N - 1) - 1` split nodes.
  Compute cost is proportional to 2^max_depth. Typical optimal values are in the 3 to 9 range.
- `min_weight=8.0`:       Minimum weight needed in a node to perform a split. Matches the number of observations by default or the sum of weights as provided by the `weights` vector. Must be > 0.
- `rowsample=1.0`:        Proportion of rows that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `colsample=1.0`:        Proportion of columns / features that are sampled at each iteration to build the tree. Should be in `]0, 1]`.
- `nbins=64`:             Number of bins into which each feature is quantized. Buckets are defined based on quantiles, hence resulting in equal weight bins. Should be between 2 and 255.
- `monotone_constraints=Dict{Int, Int}()`: Specify monotonic constraints using a dict where the key is the feature index and the value the applicable constraint (-1=decreasing, 0=none, 1=increasing). 
  !Experimental feature: note that for MLE regression, constraints may not be enforced systematically.
- `tree_type=:binary`    Tree structure to be used. One of:
  - `:binary`:       Each node of a tree is grown independently. Tree are built depthwise until max depth is reach or if min weight or gain (see `gamma`) stops further node splits.  
  - `:oblivious`:    A common splitting condition is imposed to all nodes of a given depth. 
- `rng=123`:              Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`: Hardware device to use for computations. Can be either `:cpu` or `gpu`. Following losses are not GPU supported at the moment: `:logistic_mle`.

# Internal API

Do `config = EvoTreeMLE()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in EvoTreeMLE(max_depth=...).

## Training model

A model is built using [`fit_evotree`](@ref):

```julia
model = fit_evotree(config; x_train, y_train, kwargs...)
```

## Inference

Predictions are obtained using [`predict`](@ref) which returns a `Matrix` of size `[nobs, nparams]` where the second dimensions refer to `μ` & `σ` for Normal/Gaussian and `μ` & `s` for Logistic.

```julia
EvoTrees.predict(model, X)
```

Alternatively, models act as a functor, returning predictions when called as a function with features as argument:

```julia
model(X)
```

# MLJ

From MLJ, the type can be imported using:

```julia
EvoTreeMLE = @load EvoTreeMLE pkg=EvoTrees
```

Do `model = EvoTreeMLE()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoTreeMLE(loss=...)`.

## Training data

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

## Operations

- `predict(mach, Xnew)`: returns a vector of Gaussian or Logistic distributions (according to provided `loss`) given features `Xnew` having the same scitype as `X` above.
Predictions are probabilistic.

Specific metrics can also be predicted using:

  - `predict_mean(mach, Xnew)`
  - `predict_mode(mach, Xnew)`
  - `predict_median(mach, Xnew)`

## Fitted parameters

The fields of `fitted_params(mach)` are:

  - `:fitresult`: The `GBTree` object returned by EvoTrees.jl fitting algorithm.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

```
# Internal API
using EvoTrees
config = EvoTreeMLE(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = 1_000, 5
x_train, y_train = randn(nobs, nfeats), rand(nobs)
model = fit_evotree(config; x_train, y_train)
preds = EvoTrees.predict(model, x_train)
```

```
# MLJ Interface
using MLJ
EvoTreeMLE = @load EvoTreeMLE pkg=EvoTrees
model = EvoTreeMLE(max_depth=5, nbins=32, nrounds=100)
X, y = @load_boston
mach = machine(model, X, y) |> fit!
preds = predict(mach, X)
preds = predict_mean(mach, X)
preds = predict_mode(mach, X)
preds = predict_median(mach, X)
```
"""
EvoTreeMLE

# function MLJ.clean!(model::EvoTreeRegressor)
#     warning = ""
#     if model.nrounds < 1
#         warning *= "Need nrounds ≥ 1. Resetting nrounds=1. "
#         model.nrounds = 1
#     end
#     if model.lambda < 0
#         warning *= "Need lambda ≥ 0. Resetting lambda=0. "
#         model.lambda = 0.0
#     end
#     if model.gamma < 0
#         warning *= "Need gamma ≥ 0. Resetting gamma=0. "
#         model.gamma = 0.0
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
