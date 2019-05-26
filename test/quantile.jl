using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using EvoTrees: sigmoid, logit
using EvoTrees: get_gain, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, predict, predict!, find_split!, SplitTrack, update_track!
using EvoTrees: get_edges, binarize
using EvoTrees: Quantile, Linear, Logistic, Poisson, QuantileRegression, GradientRegression

# prepare a dataset
features = rand(1_000, 1)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# q50
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.5,
    nrounds=1, nbins = 100,
    λ = 0.0, γ=0.0, η=1.0,
    max_depth = 1, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
𝑤 = ones(size(X, 1))
pred = zeros(size(Y, 1))
update_grads!(params1.loss, params1.α, pred, Y, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ), sum(δ²), sum(𝑤)
gain = get_gain(params1.loss, ∑δ, ∑δ², ∑𝑤, params1.λ)
