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
    nrounds=1,
    bagging_size=16,
    eta=1.0,
    nbins=32,
    max_depth=8,
    row_sample=0.5,
    L2=0,
)

model = fit(
    config;
    x_train,
    y_train,
    print_every_n=25,
);
pred_train_linear = predict(model, x_train)


# MAE
config = EvoTreeRegressor(;
    loss=:mae,
    nrounds=1,
    bagging_size=16,
    eta=1.0,
    nbins=32,
    max_depth=8,
    row_sample=0.5,
    L2=0,
)
model = fit(
    config;
    x_train,
    y_train,
    print_every_n=25,
);
pred_train_mae = model(x_train; device=_device)

# logloss
config = EvoTreeRegressor(;
    loss=:logloss,
    nrounds=1,
    bagging_size=16,
    eta=1.0,
    nbins=32,
    max_depth=8,
    row_sample=0.5,
    L2=0,
)

model = fit(
    config;
    x_train,
    y_train,
    print_every_n=25,
);
pred_train_logistic = model(x_train; device=_device)

# logloss xgboost
using XGBoost
params_xgb = Dict(
    :num_round => 1,
    :max_depth => 8,
    :eta => 1.0,
    :objective => "reg:logistic",
    :print_every_n => 5,
    :subsample => 0.5,
    :colsample_bytree => 1.0,
    :tree_method => "hist", # hist/gpu_hist
    :max_bin => 32,
)
dtrain = DMatrix(x_train, y_train)
watchlist = Dict("train" => DMatrix(x_train, y_train))
m_xgb = xgboost(dtrain; watchlist, verbosity=0, eval_metric="mae", params_xgb...)
pred_train_xgb = XGBoost.predict(m_xgb, x_train)

###########################################
# plot
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target", xlabelsize=18, ylabelsize=18, xticklabelsize=18, yticklabelsize=18)
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#c9ccd1",
    markersize=6)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_linear[x_perm],
    color="#26a671",
    linewidth=3,
    label="mse",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_mae[x_perm],
    color="#5891d5",
    linewidth=3,
    label="mae",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_logistic[x_perm],
    color="#e5616c",
    linewidth=3,
    label="logloss",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_xgb[x_perm],
    color="#7d568a",
    linewidth=3,
    label="xgb-logistic",
)
Legend(f[2, 1], ax;
    halign=:left,
    orientation=:horizontal,
    labelsize=18
)
# Legend(f[1, 2], ax; labelsize = 18)
f
save(joinpath(@__DIR__, "bagging-limitations.png"), f)

###############################
# credibility losses
###############################
# cred_var
config = EvoTreeRegressor(;
    loss=:cred_var,
    metric=:mse,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    lambda=0.0,
    eta=1.0,
    max_depth=9,
    rowsample=0.5,
    tree_type,
    device=_device
)
model = fit(config; x_train, y_train, print_every_n=25,)
pred_train_cred_var = model(x_train; device=_device)
sqrt(mean((pred_train_cred_var .- y_train) .^ 2))

# cred_std
config = EvoTreeRegressor(;
    loss=:cred_std,
    metric=:mse,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    lambda=0.0,
    eta=1.0,
    max_depth=9,
    rowsample=0.5,
    tree_type,
    device=_device
)
model = fit(config; x_train, y_train, print_every_n=25,)
pred_train_cred_std = model(x_train; device=_device)
sqrt(mean((pred_train_cred_std .- y_train) .^ 2))

###########################################
# plot credibility
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target", xticklabelsize=18, yticklabelsize=18)
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_cred_var[x_perm],
    color="navy",
    linewidth=1,
    label="cred_var",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_cred_std[x_perm],
    color="darkred",
    linewidth=1,
    label="cred_std",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save(joinpath(@__DIR__, "bagging-limitations-cred.png"), f)
