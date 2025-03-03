
# EvoTrees <a href="https://evovest.github.io/EvoTrees.jl/dev/"><img src="docs/src/assets/logo.png" align="right" height="160"/></a>


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
| 100k     | 10         | 6              | 0.37           | 0.69           | 0.03           | 0.03           |
| 100k     | 10         | 11             | 2.65           | 1.08           | 0.07           | 0.06           |
| 100k     | 100        | 6              | 0.98           | 1.34           | 0.08           | 0.13           |
| 100k     | 100        | 11             | 14.02          | 3.39           | 0.12           | 0.17           |
| 1M       | 10         | 6              | 2.53           | 7.20           | 0.26           | 0.37           |
| 1M       | 10         | 11             | 6.95           | 8.85           | 0.81           | 0.74           |
| 1M       | 100        | 6              | 6.15           | 13.57          | 0.73           | 1.43           |
| 1M       | 100        | 11             | 28.22          | 18.77          | 1.19           | 1.64           |
| 10M      | 10         | 6              | 27.09          | 86.61          | 3.09           | 3.15           |
| 10M      | 10         | 11             | 59.62          | 112.72         | 7.07           | 6.66           |
| 10M      | 100        | 6              | 73.04          | 151.22         | 5.78           | 14.12          |
| 10M      | 100        | 11             | 196.00         | 193.70         | 11.29          | 17.63          |

### GPU:

| **nobs** | **nfeats** | **max\_depth** | **train\_evo** | **train\_xgb** | **infer\_evo** | **infer\_xgb** |
|:--------:|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 100k     | 10         | 6              | 1.32           | 0.29           | 0.01           | 0.02           |
| 100k     | 10         | 11             | 16.74          | 1.32           | 0.01           | 0.02           |
| 100k     | 100        | 6              | 1.97           | 0.63           | 0.03           | 0.14           |
| 100k     | 100        | 11             | 31.66          | 3.31           | 0.05           | 0.16           |
| 1M       | 10         | 6              | 2.39           | 0.96           | 0.05           | 0.14           |
| 1M       | 10         | 11             | 26.17          | 2.72           | 0.06           | 0.19           |
| 1M       | 100        | 6              | 3.83           | 2.92           | 0.35           | 1.35           |
| 1M       | 100        | 11             | 45.11          | 8.11           | 0.34           | 1.62           |
| 10M      | 10         | 6              | 9.63           | 7.79           | 0.54           | 1.67           |
| 10M      | 10         | 11             | 49.88          | 13.12          | 0.58           | 1.74           |
| 10M      | 100        | 6              | 23.27          | 28.22          | 3.17           | 14.57          |
| 10M      | 100        | 11             | 81.26          | 52.97          | 3.35           | 17.95          |

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

Plot a model *ith* tree (first *actual* tree is #2 as 1st *tree* is reserved to set the model's bias):

```julia
plot(m, 2)
```

![](docs/src/assets/plot_tree.png)


## Save/Load

```julia
EvoTrees.save(m, "data/model.bson")
m = EvoTrees.load("data/model.bson");
```
