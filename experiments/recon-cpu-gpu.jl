using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using DataFrames
using CategoricalArrays
using CUDA
using EvoTrees
using EvoTrees: predict, sigmoid, logit
using EvoTrees: fit, predict, sigmoid, logit

# prepare a dataset
nobs = 10_000
Random.seed!(123)
x_num = rand(nobs) .* 5
lvls = ["A", "B", "C"]
x_cat = categorical(rand(lvls, nobs), levels=lvls, ordered=false)
levels(x_cat)
isordered(x_cat)

y = sin.(x_num) .* 0.5 .+ 0.5
y = logit(y) .+ 1.0 .* (x_cat .== "B") .- 1.0 .* (x_cat .== "C") + randn(nobs)
y = sigmoid(y)
is = collect(1:nobs)
dtot = DataFrame(x_num=x_num, x_cat=x_cat, y=y)

# train-eval split
is = sample(is, length(is), replace=false)
train_size = 0.8
i_train = is[1:floor(Int, train_size * size(is, 1))]
i_eval = is[floor(Int, train_size * size(is, 1))+1:end]

dtrain = dtot[i_train, :]
deval = dtot[i_eval, :]

############################################
# cpu vs GPU num feature
############################################
config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=200,
    nbins=64,
    lambda=0.1,
    gamma=0.05,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    device=:cpu,
)
@time model = fit(
    config,
    dtrain;
    feature_names=["x_cat", "x_num"],
    target_name="y",
    deval,
    print_every_n=25,
    verbosity=0
);
@time pred_cpu = model(dtrain; device=:cpu);

config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=200,
    nbins=64,
    lambda=0.1,
    gamma=0.05,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    device=:gpu,
)
@time model = fit(
    config,
    dtrain;
    feature_names=["x_cat", "x_num"],
    target_name="y",
    deval,
    print_every_n=25,
    verbosity=0
);
@time pred_gpu = model(dtrain; device=:cpu);
cor(pred_cpu, pred_gpu) # ~0.999%

############################################
# cpu vs GPU num feature
############################################
config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=200,
    nbins=64,
    lambda=0.1,
    gamma=0.05,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    device=:cpu,
)
@time model = fit(
    config,
    dtrain;
    feature_names=["x_num"],
    target_name="y",
    deval,
    print_every_n=25,
    verbosity=0
);
@time pred_cpu = model(dtrain; device=:cpu);

config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=200,
    nbins=64,
    lambda=0.1,
    gamma=0.05,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    device=:gpu,
)
@time model = fit(
    config,
    dtrain;
    feature_names=["x_num"],
    target_name="y",
    deval,
    print_every_n=25,
    verbosity=0
);
@time pred_gpu = model(dtrain; device=:cpu);
cor(pred_cpu, pred_gpu) # ~0.9999
