using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using CairoMakie
using CUDA
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

# using ProfileView

# prepare a dataset
tree_type = :binary # binary/oblivious
_device = :cpu

Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
# Y = sin.(features) .* 0.5 .+ 0.5
# Y = logit(Y) + randn(size(Y))
# Y = sigmoid(Y)
Y = 0.5 .* features .+ randn(size(X, 1)) ./ 10 .* features
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

# mse
config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=1,
    bagging_size=1,
    early_stopping_rounds=50,
    nbins=8,
    lambda=0.0,
    L2=10,
    eta=1.0,
    max_depth=4,
    min_weight=1.0,
    rowsample=1.0,
    colsample=1.0,
    tree_type,
    device=_device
)
model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
pred_train_linear = predict(model, x_train)

###############################
# credibility losses
###############################
# credibility
config = EvoTreeRegressor(;
    loss=:cred_var,
    metric=:mse,
    nrounds=1,
    bagging_size=1,
    early_stopping_rounds=50,
    nbins=8,
    L2=100.0,
    eta=1.0,
    max_depth=4,
    min_weight=1.0,
    rowsample=1.0,
    colsample=1.0,
    tree_type,
    device=_device
)
model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
pred_train_cred_var = model(x_train; device=_device)

###########################################
# plot credibility
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_linear[x_perm],
    color="red",
    linewidth=1,
    label="mae",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_cred_var[x_perm],
    color="green",
    linewidth=1,
    label="cred",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
# save("docs/src/assets/credibility-sinus-$tree_type-$_device.png", f)
