using Revise
using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using CairoMakie
using EvoTrees
using DataFrames
using CategoricalArrays
using EvoTrees: predict, sigmoid, logit
# using ProfileView

device = :cpu

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

# linear
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
    device,
)

@time model = fit_evotree(
    config,
    dtrain;
    feature_names=["x_cat", "x_num"],
    target_name="y",
    deval,
    print_every_n=25,
    verbosity=0
);
pred = model(dtrain);

# @btime model = fit_evotree(
#     params1,
#     dtrain;
#     fnames="x_num",
#     target_name="y",
#     verbosity=0
# );
# laptop: 51.651 ms (237548 allocations: 23.94 MiB)
# plot(logger[:metrics])
# @time pred_train_linear = predict(model, x_train);
# @time pred_eval_linear = predict(model, x_eval)
# mean((pred_train_linear .- y_train) .^ 2)
# mean((pred_eval_linear .- y_eval) .^ 2)
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    dtrain.x_num,
    dtrain.y,
    color="#BBB",
    markersize=2)
dinfer = dtrain[dtrain.x_cat.=="A", :]
x_perm = sortperm(dinfer.x_num)
pred = model(dinfer)
lines!(ax,
    dinfer.x_num[x_perm],
    pred[x_perm],
    color="lightblue",
    linewidth=1,
    label="mse - A",
)
dinfer = dtrain[dtrain.x_cat.=="B", :]
pred = model(dinfer);
x_perm = sortperm(dinfer.x_num)
lines!(ax,
    dinfer.x_num[x_perm],
    pred[x_perm],
    color="blue",
    linewidth=1,
    label="mse - B",
)
dinfer = dtrain[dtrain.x_cat.=="C", :]
pred = model(dinfer);
x_perm = sortperm(dinfer.x_num)
lines!(ax,
    dinfer.x_num[x_perm],
    pred[x_perm],
    color="navy",
    linewidth=1,
    label="mse - C",
)
f
