using DataFrames
using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
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
    loss=:linear, metric=:mse,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

# for 100k: 428.281 ms (532170 allocations: 196.80 MiB)
# for 1.25e6: 6.964114 seconds (6.05 M allocations: 2.350 GiB, 2.82% gc time)
# for 1.25e6 no eval: 6.200 s (44330 allocations: 2.19 GiB)
# for 1.25e6 mse with eval data: 6.321 s (45077 allocations: 2.19 GiB)
params1.nrounds
@time model, cache = init_evotree(params1, X_train, Y_train);
@time grow_evotree!(model, cache);
@time model = fit_evotree(params1, X_train, Y_train);
@btime model = fit_evotree(params1, X_train, Y_train);

@time model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=9999, early_stopping_rounds=9999);
@btime model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=9999, early_stopping_rounds=9999);

@time model = fit_evotree(params1, X_train, Y_train, early_stopping_rounds=10);
@time model = fit_evotree(params1, X_train, Y_train, print_every_n=2);

# @time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 5);
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval);
@time pred_train = predict(model, X_train)

@code_warntype predict(model, X_train)
@time pred = zeros(SVector{1,Float64}, size(X_train, 1))
@time EvoTrees.predict!(pred, model.trees[2], X_train)

@time predict(model, X_train)
@btime pred_train = predict($model, $X_train)
mean(abs.(pred_train .- Y_train))

# logistic
params1 = EvoTreeRegressor(
    loss=:logistic, metric=:logloss,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=2)
@time pred_train = predict(model, X_train)

# Quantile
params1 = EvoTreeRegressor(
    loss=:quantile, metric=:quantile, α=0.80,
    nrounds=10,
    λ = 0.1, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=2)
@time pred_train = predict(model, X_train)

# gaussian
params1 = EvoTreeRegressor(
    loss=:gaussian, metric=:gaussian,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 10.0,
    rowsample=0.5, colsample=0.5, nbins=32)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=2)
@time pred_train = predict(model, X_train)
