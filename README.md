
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


## Installation

Latest:

```julia-repl
julia> Pkg.add(url="https://github.com/Evovest/EvoTrees.jl")
```

From General Registry:

```julia-repl
julia> Pkg.add("EvoTrees")
```

## Performance

Data consists of randomly generated float32. Training is performed on 200 iterations. Code to reproduce is [here](https://github.com/Evovest/EvoTrees.jl/blob/main/experiments/benchmarks-regressor.jl). 

EvoTrees: v0.14.0
XGBoost: v2.0.2

CPU: 12 threads on AMD Ryzen 5900X
GPU: NVIDIA RTX A4000

### Training: 

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |     1.29s    |   1.05s  |     2.96s    |
| 500K x 100          |     6.73s    |   3.15s  |     3.83s    |
| 1M x 100            |     13.27s   |   6.01s  |     4.94s    |
| 5M x 100            |     65.1s    |   34.1s  |     14.1s    |
| 10M x 100           |     142s     |   71.8s  |     25.1s    |

### Inference:

| Dimensions   / Algo | XGBoost Hist | EvoTrees | EvoTrees GPU |
|---------------------|:------------:|:--------:|:------------:|
| 100K x 100          |    0.107s    |  0.027s  |    0.008s    |
| 500K x 100          |    0.550s    |  0.209s  |    0.031s    |
| 1M x 100            |    1.10s     |  0.410s  |    0.074s    |
| 5M x 100            |    5.44s     |  2.14s   |    0.302s    |
| 10M x 100           |    10.5s     |  4.35s   |    0.591s    |


## MLJ Integration

See [official project page](https://github.com/alan-turing-institute/MLJ.jl) for more info.


## Quick start with internal API

A model configuration must first be defined, using one of the model constructor:
- `EvoTreeRegressor`
- `EvoTreeClassifier`
- `EvoTreeCount`
- `EvoTreeMLE`

Model training is performed using `fit_evotree`. This function supports additional arguments to allowing to track out of sample metric and perform early stopping. Look at the docs for more details on available hyper-parameters for each of the above constructors and other options for training.

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
