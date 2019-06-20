using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Revise
using EvoTrees

# prepare a dataset
features = rand(100_000, 100)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# train model
params1 = EvoTreeRegressor(
    loss=:linear, metric=:mae,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=1.0, colsample=1.0, nbins=20)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 1)
@time pred_train = predict(model, X_train)
mean(abs.(pred_train .- Y_train))

# train model
params1 = EvoTreeRegressor(
    loss=:logistic,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=1.0, colsample=1.0, nbins=50)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=1, metric = :logloss)
@time pred_train = predict(model, X_train)
