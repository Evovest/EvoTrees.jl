using Revise
using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
# using Plots
using EvoTrees
using EvoTrees: predict, sigmoid, logit
# using ProfileView

tree_type = "binary"

# prepare a dataset
Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
is = sample(is, length(is), replace=false)
train_size = 0.8
i_train = is[1:floor(Int, train_size * size(is, 1))]
i_eval = is[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

# linear
params1 = EvoTreeRegressor(;
    loss=:mse,
    nrounds=100,
    nbins=64,
    lambda=0.01,
    gamma=0.1,
    eta=0.1,
    max_depth=5,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    rng=122
)

@time model = fit_evotree(
    params1;
    x_train,
    y_train,
);
