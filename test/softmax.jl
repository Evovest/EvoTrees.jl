using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample, quantile
using Plots
using Plots: colormap

using Revise
using EvoTrees
using EvoTrees: sigmoid, logit

# prepare a dataset
iris = CSV.read("./data/iris.csv")
names(iris)

features = iris[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]]
X = convert(Matrix, features)
Y = iris[:Species]
values = sort(unique(Y))
dict = Dict{String, Int}(values[i] => i for i in 1:length(values))
Y = map((x) -> dict[x], Y)

# train-eval split
𝑖 = collect(1:size(X,1))
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

scatter(X_train[:,1], X_train[:,2], color=Y_train, legend=nothing)
scatter(X_eval[:,1], X_eval[:,2], color=Y_eval, legend=nothing)

# linear
params1 = EvoTreeRegressor(
    loss=:softmax, metric=:mlogloss,
    nrounds=100, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
@time pred_train_linear = predict(model, X_train)
@time pred_eval_linear = predict(model, X_eval)
mean(abs.(pred_train_linear .- Y_train))
sqrt(mean((pred_train_linear .- Y_train) .^ 2))
