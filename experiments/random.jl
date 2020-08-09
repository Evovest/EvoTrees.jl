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
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

# for 100k 10 rounds: 410.477 ms (44032 allocations: 182.68 MiB)
# for 100k 100 rounds: 2.177 s (404031 allocations: 626.45 MiB)
# for 1.25e6 no eval: 6.244 s (73955 allocations: 2.18 GiB)
# for 1.25e6 mse with eval data:  6.345 s (74009 allocations: 2.18 GiB)
@time model = fit_evotree(params1, X_train, Y_train);
@btime model = fit_evotree(params1, X_train, Y_train);
@time pred_train = predict(model, X_train)
@time gain = importance(model, 1:100)


################################
# GPU
################################
# train model
params1 = EvoTreeRegressor(
    loss=:linear, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

@time model, cache = EvoTrees.init_evotree_gpu(params1, X_train, Y_train);
@time EvoTrees.grow_evotree_gpu!(model, cache);

@time model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
# X_train_32 = Float32.(X_train)
@time pred_train = EvoTrees.predict_gpu(model, X_train)
mean(pred_train)


# xgboost benchmark
using XGBoost
num_round = 100
param = ["max_depth" => 5,
         "eta" => 0.05,
         "objective" => "reg:squarederror",
         "print_every_n" => 5,
         "subsample" => 0.5,
         "colsample_bytree" => 0.5,
         "tree_method" => "hist",
         "nthread" => 16,
         "max_bin" => 32]
metrics = ["rmse"]
@time xgboost(X_train, num_round, label = Y_train, param = param, metrics=metrics, silent=1);
@time dtrain = DMatrix(X_train, label = Y_train)
@time model_xgb = xgboost(dtrain, num_round, param = param, silent=1);
@btime model_xgb = xgboost(dtrain, num_round, param = param, silent=1);
@time pred_train = XGBoost.predict(model_xgb, X_train)

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
    nrounds=100,
    λ = 0.0f0, γ=0.0f0, η=0.1f0,
    max_depth = 6, min_weight = 1.0f0,
    rowsample=0.5f0, colsample=0.5f0, α=0.5f0, nbins=32)
@time model = fit_evotree(params1, X_train, Y_train);
@time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n=10)
@time pred_train = predict(model, X_train)

# Quantile
params1 = EvoTreeRegressor(
    loss=:quantile, metric=:quantile, α=0.80f0,
    nrounds=100,
    λ = 0.1f0, γ=0.0f0, η=0.1f0,
    max_depth = 6, min_weight = 1.0f0,
    rowsample=0.5f0, colsample=0.5f0, nbins=32)
@time model = fit_evotree(params1, X_train, Y_train);
@time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n=10)
@time pred_train = predict(model, X_train)

# gaussian
params1 = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=100, α=0.5f0,
    λ = 0.0f0, γ=0.0f0, η=0.1f0,
    max_depth = 6, min_weight = 10.0f0,
    rowsample=0.5f0, colsample=0.5f0, nbins=32)
@time model = fit_evotree(params1, X_train, Y_train);
@time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n=10)
@time pred_train = predict(model, X_train)

# softmax
params1 = EvoTreeClassifier(
    loss=:softmax, metric=:mlogloss,
    nrounds=100, α=0.5f0,
    λ=0.0f0, γ=0.0f0, η=0.1f0,
    max_depth = 6, min_weight = 10.0f0,
    rowsample=0.5f0, colsample=0.5f0, nbins=32)

Y_train_int = UInt32.(round.(Y_train*2) .+ 1)
Y_eval_int = UInt32.(round.(Y_eval*2) .+ 1)
Y_train_int = Int.(Y_train_int)
@time model = fit_evotree(params1, X_train, Y_train_int, print_every_n=10);
@time model = fit_evotree(params1, X_train, Y_train_int, X_eval = X_eval, Y_eval = Y_eval_int, print_every_n=10)
@time pred_train = predict(model, X_train)
