using Statistics
using StatsBase:sample
using EvoTrees
using BenchmarkTools
# using BSON: @save, @load
using Plots
using NetworkLayout

# prepare a dataset
features = rand(Int(1.25e5), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

#############################
# CPU - linear
#############################
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=10,
    λ=1.0, γ=0.5, η=0.2,
    max_depth=5, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=32,
    device="cpu")

# for 100k 10 rounds: 410.477 ms (44032 allocations: 182.68 MiB)
# for 100k 100 rounds: 2.177 s (404031 allocations: 626.45 MiB)
# for 1.25e6 no eval: 6.244 s (73955 allocations: 2.18 GiB)
# for 1.25e6 mse with eval data:  6.345 s (74009 allocations: 2.18 GiB)
@time model = fit_evotree(params1, X_train, Y_train);
EvoTrees.save(model, "data/model_linear_cpu.bson")
plot(model, 2)

params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=10,
    λ=1.0, γ=0.5, η=0.2,
    max_depth=5, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=32,
    device="gpu")

# for 100k 10 rounds: 410.477 ms (44032 allocations: 182.68 MiB)
# for 100k 100 rounds: 2.177 s (404031 allocations: 626.45 MiB)
# for 1.25e6 no eval: 6.244 s (73955 allocations: 2.18 GiB)
# for 1.25e6 mse with eval data:  6.345 s (74009 allocations: 2.18 GiB)
@time model = fit_evotree(params1, X_train, Y_train);
EvoTrees.save(model, "data/model_linear_gpu.bson")

#############################
# CPU - Gaussian
#############################
params1 = EvoTreeGaussian(T=Float64,
    loss=:gaussian, metric=:none,
    nrounds=10,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=5, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=64,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train);
EvoTrees.save(model, "data/model_gaussian_gpu.bson")
