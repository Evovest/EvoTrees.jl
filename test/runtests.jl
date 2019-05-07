using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Test

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

# set parameters
loss = :linear
nrounds = 200
λ = 0.5
γ = 0.5
η = 0.05
max_depth = 5
min_weight = 1.0
rowsample = 0.5
colsample = 1.0
nbins = 250

# linear
params1 = Params(:linear, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:mae)
@time pred_train_linear = predict(model, X_train)
sqrt(mean((pred_train_linear .- Y_train) .^ 2))

# logistic / cross-entropy
params1 = Params(:logistic, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_logistic = predict(model, X_train)
sqrt(mean((pred_train_logistic .- Y_train) .^ 2))

# Poisson
params1 = Params(:poisson, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_poisson = predict(model, X_train)
sqrt(mean((pred_train_poisson .- Y_train) .^ 2))
