using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]

#############################
# CPU - linear
#############################
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:mse,
    nrounds=100,
    lambda=1.0, gamma=0, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=64,
    device=:cpu)

# asus laptopt: for 1.25e6 no eval: 9.650007 seconds (893.53 k allocations: 2.391 GiB, 5.52% gc time)
@time model = fit_evotree(params1; x_train, y_train);
@time model = fit_evotree(params1; x_train, y_train, metric=:mse, x_eval, y_eval, print_every_n=20, verbosity=1);
@time pred_train = predict(model, x_train);
gain = importance(model)

@time model, cache = EvoTrees.init(params1, x_train, y_train);
@time EvoTrees.grow_evotree!(model, cache, params1);

x1, x2 = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask,
    cache.is_in_p, cache.is_out_p, cache.mask_p,
    params1.rowsample, params1.rng)
