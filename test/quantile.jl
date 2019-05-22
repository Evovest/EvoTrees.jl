using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using EvoTrees: sigmoid, logit
using EvoTrees: get_gain, get_gain_q, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, update_track_q!
using EvoTrees: get_edges, binarize
using EvoTrees: Quantile, Linear, Logistic, Poisson, QuantileRegression, GradientRegression

# prepare a dataset
features = rand(100, 10)
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
nbins = 8
α = 8

params1 = EvoTreeRegressor(loss=:quantile, nrounds = 1, α=0.5)

# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
𝑤 = ones(size(X, 1))
pred = zeros(size(Y, 1))
# @time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
update_grads!(Val{params1.loss}(), params1.α, pred, Y, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ), sum(δ²), sum(𝑤)
gain = get_gain(∑δ, ∑δ², ∑𝑤, params1.λ)
gain = get_gain_q(∑δ, ∑δ², ∑𝑤, params1.λ)


# Calculate the gain for a given split
function bonjour(loss::T, x) where {T<:GradientRegression}
    x = x^2
    return x
end
function bonjour(loss::T, x) where {T<:QuantileRegression}
    x = x^3
    return x
end

loss = :quantile
if loss == :linear model_type = Linear()
elseif loss == :poisson model_type = Poisson()
elseif loss == :quantile model_type = Quantile()
elseif loss == :logistic model_type = Logistic()
end
