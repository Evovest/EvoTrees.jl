using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools

# prepare a dataset
X = rand(Int(2.e6), 100)
Y = rand(size(X, 1))

#######################
# EvoTrees
#######################

config = EvoTreeRegressor(
    loss=:linear, metric=:none,
    nrounds=100,
    λ = 0.0, γ=0.0, η=0.05,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

@time model = fit_evotree(config, X, Y);
@time pred = EvoTrees.predict(model, X)

#######################
# xgboost
#######################
num_round = 100
param = ["max_depth" => 5,
         "eta" => 0.05,
         "objective" => "reg:linear",
         "print_every_n" => 5,
         "subsample" => 0.5,
         "colsample_bytree" => 0.5,
         "tree_method" => "hist",
         "max_bin" => 32]
metrics = ["rmse"]

@time model_xgb = xgboost(X, num_round, label = Y, param = param, silent=1);
@time pred = XGBoost.predict(model_xgb, X)
