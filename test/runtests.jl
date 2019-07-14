using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Test
using Revise
using EvoTrees
using EvoTrees: sigmoid, logit

# prepare a dataset
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))
seed = 127
# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(
    loss=:linear, metric=:mse,
    nrounds=100, nbins=100,
    λ = 0.5, γ=0.1, η=0.01,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed = seed)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_linear = EvoTrees.predict(model, X_train)

@time p1 = EvoTrees.predict(model, X_eval)
mean(abs.(p1 - Y_eval))

# continue training
@time model2 = grow_gbtree!(model, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_linear2 = EvoTrees.predict(model2, X_train)


# logistic / cross-entropy
params1 = EvoTreeRegressor(
    loss=:logistic, metric = :logloss,
    nrounds=100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed = seed)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_logistic = EvoTrees.predict(model, X_train)

# Poisson
params1 = EvoTreeRegressor(
    loss=:poisson, metric = :logloss,
    nrounds=100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed = seed)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_poisson = EvoTrees.predict(model, X_train)

params1 = EvoTreeRegressor(
    loss=:L1, α=0.5, metric = :mae,
    nrounds=100, nbins=100,
    λ = 0.5, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed = seed)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_L1 = predict(model, X_train)

# Quantiles
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.5, metric = :quantile,
    nrounds=100,
    λ = 0.5, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed = seed)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_poisson = EvoTrees.predict(model, X_train)
