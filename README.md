
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
Code to reproduce is available in [`benchmarks/regressor.jl`](https://github.com/Evovest/EvoTrees.jl/blob/main/benchmarks/regressor.jl). 

- Run Environment:
    - CPU: 12 threads on AMD Ryzen 5900X
    - GPU: NVIDIA RTX A4000
    - Julia: v1.10.8
- Algorithms
    - XGBoost: v2.5.1 (Using the `hist` algorithm)
    - EvoTrees: v0.17.0

### CPU:

| **nobs** | **nfeats** | **max\_depth** | **train\_evo** | **train\_xgb** | **infer\_evo** | **infer\_xgb** |
|:--------:|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 100k     | 10         | 6              | 0.4            | 0.7            | 0.0            | 0.0            |
| 100k     | 10         | 11             | 5.8            | 1.1            | 0.1            | 0.1            |
| 100k     | 100        | 6              | 1.2            | 1.4            | 0.1            | 0.1            |
| 100k     | 100        | 11             | 18.3           | 3.5            | 0.1            | 0.2            |
| 1M       | 10         | 6              | 2.5            | 6.3            | 0.3            | 0.3            |
| 1M       | 10         | 11             | 11.6           | 8.0            | 0.7            | 0.6            |
| 1M       | 100        | 6              | 6.5            | 14.7           | 0.7            | 1.3            |
| 1M       | 100        | 11             | 33.4           | 19.0           | 1.2            | 1.7            |
| 10M      | 10         | 6              | 28.6           | 86.7           | 3.9            | 2.9            |
| 10M      | 10         | 11             | 66.6           | 113.0          | 6.9            | 6.3            |
| 10M      | 100        | 6              | 74.2           | 151.0          | 6.6            | 14.2           |
| 10M      | 100        | 11             | 198.0          | 192.0          | 12.2           | 17.8           |

### GPU:

| **nobs** | **nfeats** | **max\_depth** | **train\_evo** | **train\_xgb** | **infer\_evo** | **infer\_xgb** |
|:--------:|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 100k     | 10         | 6              | 1.14           | 0.28           | 0.01           | 0.02           |
| 100k     | 10         | 11             | 17.56          | 1.29           | 0.01           | 0.02           |
| 100k     | 100        | 6              | 1.75           | 0.61           | 0.04           | 0.14           |
| 100k     | 100        | 11             | 32.62          | 3.21           | 0.04           | 0.17           |
| 1M       | 10         | 6              | 2.27           | 0.96           | 0.05           | 0.15           |
| 1M       | 10         | 11             | 27.10          | 2.73           | 0.06           | 0.19           |
| 1M       | 100        | 6              | 3.71           | 2.89           | 0.35           | 1.37           |
| 1M       | 100        | 11             | 45.50          | 7.90           | 0.37           | 1.63           |
| 10M      | 10         | 6              | 9.11           | 7.46           | 0.53           | 1.73           |
| 10M      | 10         | 11             | 46.86          | 13.13          | 0.59           | 1.76           |
| 10M      | 100        | 6              | 22.74          | 28.32          | 3.43           | 14.77          |
| 10M      | 100        | 11             | 80.63          | 52.68          | 3.50           | 17.88          |


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

### DataFrames input

When using a DataFrames as input, features with elements types `Real` (incl. `Bool`) and `Categorical` are automatically recognized as input features. Alternatively, `fnames` kwarg can be used to specify the variables to be used as features. 

`Categorical` features are treated accordingly by the algorithm: ordered variables are treated as numerical features, using `≤` split rule, while unordered variables are using `==`. Support is currently limited to a maximum of 255 levels. `Bool` variables are treated as unordered, 2-levels categorical variables.

```julia
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
m = fit(config, dtrain; target_name="y");
m = fit(config, dtrain; target_name="y", fnames=["x1", "x3"]);
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

![](docs/src/assets/plot_tree.png)

Note that 1st tree is used to set the bias so the first real tree is #2.

## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```
