using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using EvoTrees
using EvoTrees: sigmoid, logit
using Serialization

# prepare a dataset
Random.seed!(12)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(
    T = Float64,
    loss = :linear,
    metric = :mse,
    nrounds = 200,
    nbins = 64,
    lambda = 0.1,
    gamma = 0.1,
    eta = 0.05,
    max_depth = 6,
    min_weight = 1.0,
    rowsample = 0.5,
    colsample = 1.0,
    rng = 123,
)

m = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n = 25);
p = m(x_eval)

# serialize(joinpath(@__DIR__, "..", "data", "save-load-test-m-v182.dat"), m);
# serialize(joinpath(@__DIR__, "..", "data", "save-load-test-p-v182.dat"), p);

# m_172 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-m-v172.dat"));
# p_172 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-p-v172.dat"));
# pm_172 = m_172(x_eval)

# m_180 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-m-v180.dat"));
# p_180 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-p-v180.dat"));
# pm_180 = m_180(x_eval)

# m_182 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-m-v182.dat"));
# p_182 = deserialize(joinpath(@__DIR__, "..", "data", "save-load-test-p-v182.dat"));
# pm_182 = m_182(x_eval)

# @assert all(p .== p_172)
# @assert all(p .== pm_172)
# @assert all(p .== p_180)
# @assert all(p .== pm_180)
# @assert all(p .== p_182)
# @assert all(p .== pm_182)

# @info "test successful! 🚀"