# EvoTrees.jl

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

Then fitting can be performed using [`EvoTrees.fit`](@ref). 2 broad methods are supported: Matrix and Tables based inputs. Optional kwargs can be used to specify eval data on which to track eval metric and perform early stopping. Look at the docs for more details on available hyper-parameters for each of the above constructors and other options for training.

Predictions are obtained by passing features data to the model. Model acts as a functor, ie. it's a struct containing the fitted model as well as a function generating the prediction of that model for the features argument. 


### Tables and DataFrames input

When using a `Tables` compatible input such as `DataFrames`, features with element type `Real` (incl. `Bool`) and `Categorical` are automatically recognized as input features. Alternatively, `feature_names` kwarg can be used. 

`Categorical` features are treated accordingly by the algorithm. Ordered variables will be treated as numerical features, using `≤` split rule, while unordered variables are using `==`. Support is currently limited to a maximum of 255 levels. `Bool` variables are treated as unordered, 2-levels cat variables.

```julia
using EvoTrees
using EvoTrees: fit
using DataFrames

config = EvoTreeRegressor(
    loss=:mse, 
    nrounds=100, 
    max_depth=6,
    nbins=32,
    eta=0.1)

x_train, y_train = rand(1_000, 10), rand(1_000)
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
m = fit(config, dtrain; target_name="y");
m = fit(config, dtrain; target_name="y", feature_names=["x1", "x3"]); # to only use specified features
preds = m(dtrain)
```

### Matrix features input

```julia
using EvoTrees
using EvoTrees: fit

config = EvoTreeRegressor(
    loss=:mse, 
    nrounds=100, 
    max_depth=6,
    nbins=32,
    eta=0.1)

x_train, y_train = rand(1_000, 10), rand(1_000)
m = fit(config; x_train, y_train)
preds = m(x_train)
```

### GPU Acceleration

EvoTrees supports training and inference on Nvidia GPU's with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
Note that on Julia ≥ 1.9 CUDA support is only enabled when CUDA.jl is installed and loaded, by another package or explicitly with e.g.
```julia
using CUDA
```

If running on a CUDA enabled machine, training and inference on GPU can be triggered through the `device` kwarg passed to the learner's constructor: 

```julia
config = EvoTreeRegressor(
    loss=:mse, 
    device=:gpu
)

m = fit(config, dtrain; target_name="y");
p = m(dtrain; device=:gpu)
```

## Reproducibility

EvoTrees models trained on cpu can be fully reproducible.

Models of the gradient boosting family typically involve some stochasticity. 
In EvoTrees, this primarily concern the the 2 subsampling parameters `rowsample` and `colsample`. The other stochastic operation happens at model initialisation when the features are binarized to allow for fast histogram construction: a random subsample of `1_000 * nbins` is used to compute the breaking points. 

These random parts of the algorithm can be deterministically reproduced on cpu by specifying a `seed` to the model constructor. `seed` is an integer, which defaults to `123`. 
A `Random.Xoshiro` generator will be created at training's initialization and stored in cache to provide a consistent, reproducible random number generation.  

Consequently, the following `m1` and `m2` models will be identical:

```julia
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m1 = fit(config, dtrain; target_name="y");
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m2 = fit(config, dtrain; target_name="y");
```

Since the random generator is initialized within the `fit` step, the following `m1` and `m2` models will also be identical:

```julia
config = EvoTreeRegressor(rowsample=0.5, rng=123)
m1 = fit(config, dtrain; target_name="y");
m2 = fit(config, dtrain; target_name="y");
```

Note that in presence of multiple identical or very highly correlated features, model may not be reproducible if features are permuted since in situation where 2 features provide identical gains, the first one will be selected. Therefore, if the identity relationship doesn't hold on new data, different predictions will be returned from models trained on different features order. 

Reproducibility is supported both for `cpu` and `gpu`(CUDA) devices. 

## Missing values

### Features

EvoTrees does not handle features having missing values. Proper preprocessing of the data is therefore needed (and a general good practice regardless of the ML model used).

This includes situations where values may be all non-missing, but where the `eltype` is `Union{Missing,Float64}` or `Any` for example. A conversion using `identity` is then recommended: 

```julia
julia> x = Vector{Union{Missing, Float64}}([1, 2])
2-element Vector{Union{Missing, Float64}}:
 1.0
 2.0

julia> identity.(x)
2-element Vector{Float64}:
 1.0
 2.0
```

For dealing with numerical or ordered categorical features containing missing values, a common approach is to first create an `Bool` variable capturing the info on whether a value is missing:

```julia
using DataFrames
transform!(df, :my_feat => ByRow(ismissing) => :my_feat_ismissing)
```

Then, the missing values can be imputed (replaced by some default values such as `mean` or `median`, or using a more sophisticated approach such as predictions from another model):

```julia
transform!(df, :my_feat => (x -> coalesce.(x, median(skipmissing(x)))) => :my_feat)
```

For unordered categorical variables, a recode of the missing into a non missing level is sufficient:
```julia
using CategoricalArrays
julia> x = categorical(["a", "b", missing])
3-element CategoricalArray{Union{Missing, String},1,UInt32}:
 "a"
 "b"
 missing

julia> x = recode(x, missing => "missing value")
3-element CategoricalArray{String,1,UInt32}:
 "a"
 "b"
 "missing value"
```

### Target

Target variable must have its element type `<:Real`. Only exception is for `EvoTreeClassifier` for which `CategoricalValue`, `Integer`, `String` and `Char` are supported.

## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```
