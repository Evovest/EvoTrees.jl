using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Revise
using EvoTrees

# prepare a dataset
features = rand(10000) .* 20 .- 10
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
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
nrounds = 1
λ = 1.0
γ = 1e-15
η = 0.5
max_depth = 5
min_weight = 5.0
rowsample = 1.0
colsample = 1.0

# params1 = Params(nrounds, λ, γ, η, max_depth, min_weight, :linear)
params1 = Params(:linear, 1, λ, γ, 1.0, 5, min_weight, rowsample, colsample)

# train model
params1 = Params(:linear, 100, 0.0, 0.0, 0.1, 5, 1.0, 0.5, 1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:none)
@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))

# train model
params1 = Params(:logistic, 100, 0.0, 0.0, 0.1, 5, 1.0, 0.5, 1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=10, metric = :logloss)
@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))
