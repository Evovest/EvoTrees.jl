# EvoTrees

[![Build Status](https://travis-ci.org/Evovest/EvoTrees.jl.svg?branch=master)](https://travis-ci.org/Evovest/EvoTrees.jl)

A Julia implementation of boosted trees.
Efficient histogram based algorithm with support for multiple loss functions (notably multi-target objectives such as max likelihood methods).

[R binding available](https://github.com/Evovest/EvoTrees)

Currently supports:

- linear
- logistic
- Poisson
- L1 (mae regression)
- Quantile
- multiclassification (softmax)
- Gaussian (max likelihood)

Input features is expected to be `Matrix{Float64/Float32}`. User friendly format conversion to be done (or see integration with MLJ).

## GPU

An experimental GPU support is now provided for linear, logistic and Gaussian objective functions. Speedup compared to multi-threaded cpu histogram is modest at the moment (~25% vs 16 CPU threads on RTX2080).

Simply call `fit_evotree_gpu()` instead of `fit_evotree()` and `predict_gpu()` instead of `predict()`.

## Installation

Latest:

```julia-repl
julia> Pkg.add("https://github.com/Evovest/EvoTrees.jl")
```

Official Repo:

```julia-repl
julia> Pkg.add("EvoTrees")
```


## Performance


Data consista of randomly generated flots. Training is performed on 200 iterations. Code to repduce is [here]([Benchmark](https://github.com/Evovest/EvoTrees.jl/blob/master/blog/benchmarks_v2.jl)). 

EvoTrees: v0.7.0
XGBoost: v1.1.1

CPU: AMD Ryzen 7 4800H (16 threads)
GPU: NVIDIA GTX 1660 (6 gb )


| Dimensions / Algo | XGBoost Exact | XGBoost Hist  | EvoTrees CPU | EvoTrees GPU |
|-------------------|:-------------:|:-------------:|:------------:|:------------:|
| 10K x 100         |       -       |     0.83s     |     0.40s    |     4.18s    |
| 100K x 100        |       -       |     1.96s     |     1.89s    |     5.19s    |
| 500K x 100        |       -       |     7.13s     |     9.91s    |     9.44s    |
| 1M X 100          |     215.9s    |     13.5s     |     19.9s    |     12.7s    |
| 5M X 100          |       -       |     86.5s     |     198.6s   |     64.4s    |


## Parameters

  - loss: {:linear, :logistic, :poisson, :L1, :quantile, :softmax, :gaussian}
  - nrounds: 10L
  - λ: 0.0
  - γ: 0.0
  - η: 0.1
  - max\_depth: integer, default 5L
  - min\_weight: float \>= 0 default=1.0,
  - rowsample: float \[0,1\] default=1.0
  - colsample: float \[0,1\] default=1.0
  - nbins: Int, number of bins into which features will be quantilized default=64
  - α: float \[0,1\], set the quantile or bias in L1 default=0.5
  - metric: {:mse, :rmse, :mae, :logloss, :quantile},  default=:none


## MLJ Integration

See [official project page](https://github.com/alan-turing-institute/MLJ.jl) for more info.

```julia
using StatsBase: sample
using EvoTrees
using EvoTrees: sigmoid, logit
using MLJBase

features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = MLJBase.table(X)

# @load EvoTreeRegressor
# linear regression
tree_model = EvoTreeRegressor(loss=:linear, max_depth=5, η=0.05, nrounds=10)

# set machine
mach = machine(tree_model, X, y)

# partition data
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split

# fit data
fit!(mach, rows=train, verbosity=1)

# continue training
mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

# predict on train data
pred_train = predict(mach, selectrows(X,train))
mean(abs.(pred_train - selectrows(Y,train)))

# predict on test data
pred_test = predict(mach, selectrows(X,test))
mean(abs.(pred_test - selectrows(Y,test)))
```


## Getting started using internal API

Minimal example to fit a noisy sinus wave.

![](figures/regression_sinus.png)

```julia
using EvoTrees
using EvoTrees: sigmoid, logit

# prepare a dataset
features = rand(10000) .* 20 .- 10
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

params1 = EvoTreeRegressor(
    loss=:linear, metric=:mse,
    nrounds=100, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_eval_linear = predict(model, X_eval)

# logistic / cross-entropy
params1 = EvoTreeRegressor(
    loss=:logistic, metric = :logloss,
    nrounds=100, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_eval_logistic = predict(model, X_eval)

# Poisson
params1 = EvoTreeCount(
    loss=:poisson, metric = :poisson,
    nrounds=100, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_eval_poisson = predict(model, X_eval)

# L1
params1 = EvoTreeRegressor(
    loss=:L1, α=0.5, metric = :mae,
    nrounds=100, nbins=100,
    λ = 0.5, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_eval_L1 = predict(model, X_eval)
```

## Quantile Regression

![](figures/quantiles_sinus.png)

```julia
# q50
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.5, metric = :quantile,
    nrounds=200, nbins = 100,
    λ = 0.1, γ=0.0, η=0.05,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_train_q50 = predict(model, X_train)

# q20
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.2, metric = :quantile,
    nrounds=200, nbins = 100,
    λ = 0.1, γ=0.0, η=0.05,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_train_q20 = predict(model, X_train)

# q80
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.8,
    nrounds=200, nbins = 100,
    λ = 0.1, γ=0.0, η=0.05,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
pred_train_q80 = predict(model, X_train)
```

## Gaussian Max Likelihood

![](figures/gaussian_sinus.png)

```julia
params1 = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=100, nbins=100,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed=123)
```

## Feature importance

Returns the normalized gain by feature.

```julia
features_gain = importance(model, var_names)
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