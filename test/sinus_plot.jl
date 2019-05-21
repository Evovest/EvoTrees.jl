using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Test
using Plots

using Revise
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
η = 0.1
max_depth = 5
min_weight = 1.0
rowsample = 0.5
colsample = 1.0
nbins = 64

# linear
params1 = Params(:linear, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:mae)
@btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 10, metric=:mae)
@time pred_train_linear = predict(model, X_train)
@time pred_eval_linear = predict(model, X_eval)
sqrt(mean((pred_train_linear .- Y_train) .^ 2))

# logistic / cross-entropy
params1 = Params(:logistic, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_logistic = predict(model, X_train)
@time pred_eval_logistic = predict(model, X_eval)
sqrt(mean((pred_train_logistic .- Y_train) .^ 2))

# Poisson
params1 = Params(:poisson, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_poisson = predict(model, X_train)
@time pred_eval_poisson = predict(model, X_eval)
sqrt(mean((pred_train_poisson .- Y_train) .^ 2))

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "gray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
plot!(X_train[:,1][x_perm], pred_train_logistic[x_perm], color = "darkred", linewidth = 1.5, label = "Logistic")
plot!(X_train[:,1][x_perm], pred_train_poisson[x_perm], color = "green", linewidth = 1.5, label = "Poisson")

savefig("regression_sinus.png")

x_perm = sortperm(X_eval[:,1])
plot(X_eval, Y_eval, ms = 1, mcolor = "gray", mscolor = "gray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_eval[:,1][x_perm], pred_eval_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
plot!(X_eval[:,1][x_perm], pred_eval_logistic[x_perm], color = "darkred", linewidth = 1.5, label = "Logistic")
plot!(X_eval[:,1][x_perm], pred_eval_poisson[x_perm], color = "green", linewidth = 1.5, label = "Poisson")
