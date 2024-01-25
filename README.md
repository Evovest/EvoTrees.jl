
# EvoTrees <a href="https://evovest.github.io/EvoTrees.jl/dev/"><img src="figures/hex-evotrees-2.png" align="right" height="160"/></a>


| Documentation | CI Status | DOI |
|:------------------------:|:----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] | [![][DOI-img]][DOI-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://evovest.github.io/EvoTrees.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://evovest.github.io/EvoTrees.jl/stable

[ci-img]: https://github.com/Evovest/EvoTrees.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Evovest/EvoTrees.jl/actions?query=workflow%3ACI+branch%3Amain

[DOI-img]: https://zenodo.org/badge/164559537.svg
[DOI-url]: https://zenodo.org/doi/10.5281/zenodo.10569604

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
    - EvoTrees: v0.15.2.

### Training: 

| Dimensions   / Algo | XGBoost CPU | EvoTrees CPU | XGBoost GPU | EvoTrees GPU |
|---------------------|:-----------:|:------------:|:-----------:|:------------:|
| 100K x 100          |    2.34s    |     1.01s    |    0.90s    |     2.61s    |
| 500K x 100          |    10.7s    |     3.95s    |    1.84s    |     3.41s    |
| 1M x 100            |    21.1s    |     6.57s    |    3.10s    |     4.47s    |
| 5M x 100            |    108s     |     36.1s    |    12.9s    |     12.5s    |
| 10M x 100           |    218s     |     72.6s    |    25.5s    |     23.0s    |

### Inference:

| Dimensions   / Algo | XGBoost CPU  | EvoTrees CPU | XGBoost GPU | EvoTrees GPU |
|---------------------|:------------:|:------------:|:-----------:|:------------:|
| 100K x 100          |    0.151s    |    0.058s    |     NA      |    0.045s    |
| 500K x 100          |    0.647s    |    0.248s    |     NA      |    0.172s    |
| 1M x 100            |    1.26s     |    0.573s    |     NA      |    0.327s    |
| 5M x 100            |    6.04s     |    2.87s     |     NA      |    1.66s     |
| 10M x 100           |    12.4s     |    5.71s     |     NA      |    3.40s     |

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

When using a DataFrames as input, features with elements types `Real` (incl. `Bool`) and `Categorical` are automatically recognized as input features. Alternatively, `fnames` kwarg can be used to specify the variables to be used as features. 

`Categorical` features are treated accordingly by the algorithm: ordered variables are treated as numerical features, using `≤` split rule, while unordered variables are using `==`. Support is currently limited to a maximum of 255 levels. `Bool` variables are treated as unordered, 2-levels categorical variables.

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
