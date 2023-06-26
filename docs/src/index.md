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

Then fitting can be performed using [`fit_evotree`](@ref). This function supports additional arguments to provide eval data in order to track out of sample metrics and perform early stopping. Look at the docs for more details on available hyper-parameters for each of the above constructors and other options for training.

Predictions are obtained by passing features data to the model. Model acts as a functor, ie. it's a struct containing the fitted model as well as a function generating the prediction of that model for the features argument. 

```julia
using EvoTrees

config = EvoTreeRegressor(
    loss=:linear, 
    nrounds=100, 
    max_depth=6, 
    nbins=32,
    eta=0.1,
    lambda=0.1, 
    gamma=0.1, 
    min_weight=1.0,
    rowsample=0.5, 
    colsample=0.8)

m = fit_evotree(config; x_train, y_train)
preds = m(x_train)
```

## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```
