
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

Data consists of randomly generated float32. Training is performed on 200 iterations. Code to reproduce is [here](https://github.com/Evovest/EvoTrees.jl/blob/main/experiments/benchmarks-regressor.jl). 

EvoTrees: v0.13.1
XGBoost: v2.0.2

CPU: 12 threads on AMD Ryzen 5900X
GPU: NVIDIA RTX A4000

### Training: 

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |     1.31s    |   1.17s  |     3.20s    |
| 500K x 100          |     6.73s    |   4.77s  |     4.81s    |
| 1M x 100            |     13.27s   |   8.42s  |     6.71s    |
| 5M x 100            |     67.3s    |   43.6s  |     21.7s    |

### Inference:

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |    0.125s    |  0.030s  |    0.008s    |
| 500K x 100          |    0.550s    |  0.209s  |    0.031s    |
| 1M x 100            |    1.10s     |  0.410s  |    0.074s    |
| 5M x 100            |    5.44s     |  2.14s   |    0.302s    |


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
m = fit_evotree(config; x_train, y_train)
preds = m(x_train)
```

## Feature importance

Returns the normalized gain by feature.

```julia
features_gain = importance(m)
```

## Plot

Plot a given tree of the model:

```julia
plot(m, 2)
```

![](figures/plot_tree.png)

Note that 1st tree is used to set the bias so the first real tree is #2.

## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```

A GPU model should be converted into a CPU one before saving: `m_cpu = convert(EvoTree, m_gpu)`.