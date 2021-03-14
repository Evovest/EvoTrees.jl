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


#############################
# CPU - linear
#############################
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=64)

# Asus laptop: 12.533999 seconds (197.39 k allocations: 6.670 GiB, 9.46% gc time)
@time model = fit_evotree(params1, X_train, Y_train);
@btime model = fit_evotree($params1, $X_train, $Y_train);
@time pred_train = predict(model, X_train);
@btime pred_train = predict(model, X_train);
gain = importance(model, 1:100)

@time model, cache = EvoTrees.init_evotree(params1, X_train, Y_train);
@time EvoTrees.grow_evotree!(model, cache);

using BSON: @save, @load
@save "model.bson" model
@load "model.bson" model
pred_train = predict(model, X_train)

#############################
# CPU - Logistic
#############################
params1 = EvoTreeRegressor(T=Float32,
    loss=:logistic, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=64)
@time model = fit_evotree(params1, X_train, Y_train);
@time pred = predict(model, X_train);

#############################
# CPU - Gaussian
#############################
params1 = EvoTreeGaussian(T=Float64,
    loss=:gaussian, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

# Asus laptop: 22.040555 seconds (205.85 k allocations: 8.021 GiB, 6.17% gc time)
@time model = fit_evotree(params1, X_train, Y_train);
# Asus laptop: 1.562431 seconds (1.89 k allocations: 1.658 GiB)
@time model, cache = EvoTrees.init_evotree(params1, X_train, Y_train);

################################
# GPU - Linear
################################
# train model
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=64)

# Asus laptop:  8.898483 seconds (22.42 M allocations: 4.254 GiB, 9.09% gc time)
@time model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
@btime model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
@time model, cache = EvoTrees.init_evotree_gpu(params1, X_train, Y_train);
@time EvoTrees.grow_evotree_gpu!(model, cache);

# X_train_32 = Float32.(X_train)
@time pred_train = EvoTrees.predict_gpu(model, X_train);
@btime pred_train = EvoTrees.predict_gpu(model, X_train);
mean(pred_train)

################################
# GPU - Logistic
################################
# train model
params1 = EvoTreeRegressor(T=Float64,
    loss=:logistic, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=64)
@time model = fit_evotree_gpu(params1, X_train, Y_train);
@time pred_train = predict_gpu(model, X_train)

################################
# GPU - Gaussian
################################
params1 = EvoTreeGaussian(T=Float64,
    loss=:gaussian, metric=:none,
    nrounds=100,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)
#  Asus laptop: 10.557295 seconds (22.51 M allocations: 4.280 GiB, 5.53% gc time)
@time model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
# Asus laptop: 1.577430 seconds (9.35 k allocations: 1.613 GiB)
@time model, cache = EvoTrees.init_evotree_gpu(params1, X_train, Y_train);


############################
# xgboost
############################
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
