
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
    - EvoTrees: v0.15.2.

### CPU

| **nobs** | **nfeats** | **max\_depth** | **train\_evo** | **train\_xgb** | **infer\_evo** | **infer\_xgb** |
|:--------:|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 100k     | 10         | 6              | 0.62           | 0.69           | 0.05           | 0.03           |
| 100k     | 10         | 11             | 2.64           | 1.10           | 0.09           | 0.06           |
| 100k     | 100        | 6              | 1.93           | 1.56           | 0.07           | 0.24           |
| 100k     | 100        | 11             | 15.59          | 3.47           | 0.12           | 0.64           |
| 1M       | 10         | 6              | 5.21           | 6.66           | 0.43           | 0.36           |
| 1M       | 10         | 11             | 11.24          | 8.67           | 0.73           | 0.90           |
| 1M       | 100        | 6              | 15.45          | 14.43          | 0.59           | 1.91           |
| 1M       | 100        | 11             | 43.14          | 19.01          | 1.20           | 2.22           |
| 10M      | 10         | 6              | 46.58          | 88.28          | 3.94           | 3.56           |
| 10M      | 10         | 11             | 89.32          | 116.19         | 6.57           | 6.68           |
| 10M      | 100        | 6              | 151.54         | 149.28         | 6.43           | 17.37          |
| 10M      | 100        | 11             | 322.28         | 196.34         | 11.39          | 22.30          |

### GPU

| **nobs** | **nfeats** | **max\_depth** | **train\_evo** | **train\_xgb** | **infer\_evo** | **infer\_xgb** |
|:--------:|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| 100k     | 10         | 6              | 0.56           | 0.28           | 0.05           | 0.02           |
| 100k     | 10         | 11             | 2.87           | 1.29           | 0.08           | 0.03           |
| 100k     | 100        | 6              | 1.08           | 0.57           | 0.07           | 0.19           |
| 100k     | 100        | 11             | 5.83           | 3.21           | 0.12           | 0.20           |
| 1M       | 10         | 6              | 1.62           | 0.97           | 0.43           | 0.18           |
| 1M       | 10         | 11             | 4.76           | 2.71           | 0.70           | 0.19           |
| 1M       | 100        | 6              | 3.85           | 2.90           | 0.64           | 1.89           |
| 1M       | 100        | 11             | 11.44          | 7.99           | 1.15           | 2.40           |
| 10M      | 10         | 6              | 11.02          | 7.58           | 2.85           | 1.98           |
| 10M      | 10         | 11             | 21.62          | 13.21          | 6.78           | 2.26           |
| 10M      | 100        | 6              | 39.09          | 27.34          | 5.79           | 17.56          |
| 10M      | 100        | 11             | 81.15          | 51.02          | 11.32          | 20.28          |


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
