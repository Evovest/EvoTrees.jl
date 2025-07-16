using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using DataFrames
using Random
using CairoMakie
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

# prepare a dataset
tree_type = :binary # binary/oblivious
_device = :cpu

Random.seed!(123)
features = rand(1000) .* 4 .- 1
x_train = reshape(features, (size(features)[1], 1))
# y_train = 0.5 .* features .^ 2 # deterministic

y_train = sin.(features) .* 0.5 .+ 0.5
y_train = logit(y_train) + randn(size(y_train)) ./ 2
y_train = sigmoid(y_train)

config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=4,
    nbins=8,
    L2=0,
    eta=0.5,
    max_depth=4,
)
# _device = EvoTrees.CPU
# m, cache = EvoTrees.init(config, x_train, y_train, _device; feature_names=nothing, w_train=nothing, offset_train=nothing)
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
dtrain = Tables.columntable(dtrain)
_device = EvoTrees.CPU
m, cache = EvoTrees.init(config, dtrain, _device; target_name="y", feature_names=nothing, weight_name=nothing, offset_name=nothing)
x_bin = cache.x_bin

###########################################
# plot
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(
    f[1, 1],
    # xlabel="feature",
    # xlabelsize=18,
    ylabel="target",
    ylabelsize=18,
    xticklabelsize=18,
    yticklabelsize=18,
    title="raw",
    titlesize=16
)
scatter!(
    ax,
    x_train[:, 1],
    y_train,
    # label="raw",
    markersize=6,
    color="#5891d5"
)
f

x_perm = sortperm(x_train[:, 1])
ax = Axis(
    f[2, 1],
    xticks=0:2:12,       # show 1,3,5,7,9
    xlabel="feature",
    xlabelsize=18,
    ylabel="target",
    ylabelsize=18,
    xticklabelsize=18,
    yticklabelsize=18,
    title="binned",
    titlesize=16
)
scatter!(
    ax,
    x_bin[:, 1],
    y_train,
    # label="bins",
    markersize=6,
    color="#5891d5"
)
f
save("algo-hist.png", f)
