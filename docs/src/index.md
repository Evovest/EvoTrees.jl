# [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl)

A Julia implementation of boosted trees with CPU and GPU support. Efficient histogram based algorithms with support for multiple loss functions, including various regressions, multi-classification and Gaussian max likelihood. 

See the `examples-API` section to get started using the internal API, or `examples-MLJ` to use within the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) framework.

Complete details about hyper-parameters are found in the `Models` section.

[R binding available](https://github.com/Evovest/EvoTrees).

## Installation

Latest:

```julia-repl
julia> Pkg.add(url="https://github.com/Evovest/EvoTrees.jl")
```

From General Registry:

```julia-repl
julia> Pkg.add("EvoTrees")
```

## Quick start

A model configuration must first be defined, using one of the model constructor: 
- [`EvoTreeRegressor`](@ref)
- [`EvoTreeClassifier`](@ref)
- [`EvoTreeCount`](@ref)
- [`EvoTreeMLE`](@ref)

Then fitting can be performed using [`fit_evotree`](@ref). 2 broad methods are supported: Matrix and Tables based inputs. Optional kwargs can be used to specify eval data on which to track eval metric and perform early stopping. Look at the docs for more details on available hyper-parameters for each of the above constructors and other options for training.

Predictions are obtained by passing features data to the model. Model acts as a functor, ie. it's a struct containing the fitted model as well as a function generating the prediction of that model for the features argument. 


### Matrix features input

```julia
using EvoTrees

config = EvoTreeRegressor(
    loss=:mse, 
    nrounds=100, 
    max_depth=6,
    nbins=32,
    eta=0.1)

x_train, y_train = rand(1_000, 10), rand(1_000)
m = fit_evotree(config; x_train, y_train)
preds = m(x_train)
```

### DataFrames and Tables input

When using a Tables compatible input such as DataFrames, features with elements types `Real` (incl. `Bool`) and `Categorical` are automatically recognized as input features. Alternatively, `fnames` kwarg can be used. 

`Categorical` features are treated accordingly by the algorithm. Ordered variables will be treated as numerical features, using `≤` split rule, while unordered variables are using `==`. Support is currently limited to a maximum of 255 levels. `Bool` variables are treated as unordered, 2-levels cat variables.

```julia
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
m = fit_evotree(config, dtrain; target_name="y");
m = fit_evotree(config, dtrain; target_name="y", fnames=["x1", "x3"]);
```


### GPU Acceleration

If running on a CUDA enabled machine, training and inference on GPU can be triggered through the `device` kwarg: 

```julia
m = fit_evotree(config, dtrain; target_name="y", device="gpu");
p = m(dtrain; device="gpu")
```


## Reproducibility

EvoTrees models trained on cpu can be fully reproducible.

Models of the gradient boosting family typically involve some stochasticity. 
In EvoTrees, this primarily concern the the 2 subsampling parameters `rowsample` and `colsample`. The other stochastic operation happens at model initialisation when the features are binarized to allow for fast histogram construction: a random subsample of `1_000 * nbins` is used to compute the breaking points. 

These random parts of the algorithm can be deterministically reproduced on cpu by specifying an `rng` to the model constructor. `rng` can be an integer (ex: `123`) or a random generator (ex: `Random.Xoshiro(123)`). 
If no `rng` is specified, `123` is used by default. When an integer `rng` is used, a `Random.MersenneTwister` generator will be created by the EvoTrees's constructor. Otherwise, the provided random generator will be used.  

Consequently, the following `m1` and `m2` models will be identical:

```julia
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m1 = fit_evotree(config, df; target_name="y");
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m2 = fit_evotree(config, df; target_name="y");
```

However, the following `m1` and `m2` models won't be because the there's stochasticity involved in the model from `rowsample` and the random generator in the `config` isn't reset between the fits:

```julia
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m1 = fit_evotree(config, df; target_name="y");
m2 = fit_evotree(config, df; target_name="y");
```

Note that in presence of multiple identical or very highly correlated features, model may not be reproducible if features are permuted since in situation where 2 features provide identical gains, the first one will be selected. Therefore, if the identity relationship doesn't hold on new data, different predictions will be returned from models trained on different features order. 

At the moment, there's no reproducibility guarantee on GPU, although this may change in the future. 

## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```
