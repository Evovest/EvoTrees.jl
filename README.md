
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

Data consists of randomly generated `Matrix{Float64}`. Training is performed on 200 iterations.  
Code to reproduce is availabe in [`benchmarks/regressor.jl`](https://github.com/Evovest/EvoTrees.jl/blob/main/benchmarks/regressor.jl). 

- Run Environment:
    - CPU: 12 threads on AMD Ryzen 5900X.
    - GPU: NVIDIA RTX A4000.
    - Julia: v1.9.1.
- Algorithms
    - XGBoost: v2.3.0 (Using the `hist` algorithm).
    - EvoTrees: v0.15.0.

### Training: 

| Dimensions   / Algo | XGBoost CPU | EvoTrees CPU | XGBoost GPU | EvoTrees GPU |
|---------------------|:-----------:|:------------:|:-----------:|:------------:|
| 100K x 100          |    2.33s    |     1.09s    |    0.90s    |     2.72s    |
| 500K x 100          |    10.7s    |     2.96s    |    1.84s    |     3.65s    |
| 1M x 100            |    20.9s    |     6.48s    |    3.10s    |     4.45s    |
| 5M x 100            |    108s     |     35.8s    |    12.9s    |     12.7s    |
| 10M x 100           |    216s     |     71.6s    |    25.5s    |     23.0s    |

### Inference:

| Dimensions   / Algo | XGBoost CPU  | EvoTrees CPU | XGBoost GPU | EvoTrees GPU |
|---------------------|:------------:|:------------:|:-----------:|:------------:|
| 100K x 100          |    0.151s    |    0.053s    |     NA      |    0.036s    |
| 500K x 100          |    0.628s    |    0.276s    |     NA      |    0.169s    |
| 1M x 100            |    1.26s     |    0.558s    |     NA      |    0.334s    |
| 5M x 100            |    6.04s     |    2.87s     |     NA      |    1.66s     |
| 10M x 100           |    12.4s     |    5.71s     |     NA      |    3.31s     |

## MLJ Integration

See [official project page](https://github.com/alan-turing-institute/MLJ.jl) for more info.

## Quick start with internal API

A model configuration must first be defined, using one of the model constructor:
- `EvoTreeRegressor`
- `EvoTreeClassifier`
- `EvoTreeCount`
- `EvoTreeMLE`

Model training is performed using `fit_evotree`. 
It supports additional keyword arguments to track evaluation metric and perform early stopping. 
Look at the docs for more details on available hyper-parameters for each of the above constructors and other options training options.

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

### DataFrames input

When using a DataFrames as input, features with elements types `Real` (incl. `Bool`) and `Categorical` are automatically recognized as input features. Alternatively, `fnames` kwarg can be used. 

`Categorical` features are treated accordingly by the algorithm. Ordered variables will be treated as numerical features, using `≤` split rule, while unordered variables are using `==`. Support is currently limited to a maximum of 255 levels. `Bool` variables are treated as unordered, 2-levels cat variables.

```julia
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
m = fit_evotree(config, dtrain; target_name="y");
m = fit_evotree(config, dtrain; target_name="y", fnames=["x1", "x3"]);
```

## Feature importance

Returns the normalized gain by feature.

```julia
features_gain = EvoTrees.importance(m)
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
