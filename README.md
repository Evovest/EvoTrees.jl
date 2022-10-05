
# EvoTrees <a href="https://evovest.github.io/EvoTrees.jl/dev/"><img src="figures/hex-evotrees-2.png" align="right" height="160"/></a>


| Documentation | CI Status |
|:------------------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://evovest.github.io/EvoTrees.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://evovest.github.io/EvoTrees.jl/stable

[ci-img]: https://github.com/Evovest/EvoTrees.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Evovest/EvoTrees.jl/actions?query=workflow%3ACI+branch%3Amain

A Julia implementation of boosted trees with CPU and GPU support.
Efficient histogram based algorithms with support for multiple loss functions (notably multi-target objectives such as max likelihood methods).

[R binding available](https://github.com/Evovest/EvoTrees).

Input features are expected to be `Matrix{Float64/Float32}` when using the internal API. Tables/DataFrames format can be handled through [MLJ](https://github.com/alan-turing-institute/MLJ.jl). See the docs for further details. 


## Installation

Latest:

```julia-repl
julia> Pkg.add("https://github.com/Evovest/EvoTrees.jl")
```

From General Registry:

```julia-repl
julia> Pkg.add("EvoTrees")
```

## Performance

Data consists of randomly generated float32. Training is performed on 200 iterations. Code to reproduce is [here](https://github.com/Evovest/EvoTrees.jl/blob/master/experiments/benchmarks_v2.jl). 

EvoTrees: v0.8.4
XGBoost: v1.1.1

CPU: 16 threads on AMD Threadripper 3970X
GPU: NVIDIA RTX 2080

### Training: 

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |     1.10s    |   1.80s  |     3.14s    |
| 500K x 100          |     4.83s    |   4.98s  |     4.98s    |
| 1M x 100            |     9.84s    |   9.89s  |     7.37s    |
| 5M x 100            |     45.5s    |   53.8s  |     25.8s    |

### Inference:

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |    0.164s    |  0.026s  |    0.013s    |
| 500K x 100          |    0.796s    |  0.175s  |    0.055s    |
| 1M x 100            |     1.59s    |  0.396s  |    0.108s    |
| 5M x 100            |     7.96s    |   2.15s  |    0.543s    |


## MLJ Integration

See [official project page](https://github.com/alan-turing-institute/MLJ.jl) for more info.


## Getting started using internal API

```julia
using EvoTrees

config = EvoTreeRegressor(
    loss=:linear, 
    nrounds=100, 
    nbins=100,
    lambda=0.5, 
    gamma=0.1, 
    eta=0.1,
    max_depth=6, 
    min_weight=1.0,
    rowsample=0.5, 
    colsample=1.0)
model = fit_evotree(config; x_train = x_train, y_train = y_train)
preds = predict(model, x_eval)
```

## Feature importance

Returns the normalized gain by feature.

```julia
features_gain = importance(model)
```

## Plot

Plot a given tree of the model:

```julia
plot(model, 2)
```

![](figures/plot_tree.png)

Note that 1st tree is used to set the bias so the first real tree is #2.

## Save/Load

```julia
EvoTrees.save(model, "data/model.bson")
model = EvoTrees.load("data/model.bson");
```