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
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
# Y = 0.5 .* features .+ rand(size(X, 1))
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
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
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

# logistic / cross-entropy
config = EvoTreeRegressor(;
    loss=:logloss,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_logistic = model(x_train; device=_device)
sqrt(mean((pred_train_logistic .- y_train) .^ 2))

# poisson
config = EvoTreeCount(;
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_poisson = model(x_train; device=_device)
sqrt(mean((pred_train_poisson .- y_train) .^ 2))

# gamma
config = EvoTreeRegressor(;
    loss=:gamma,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_gamma = model(x_train; device=_device)
sqrt(mean((pred_train_gamma .- y_train) .^ 2))

# tweedie
config = EvoTreeRegressor(;
    loss=:tweedie,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_tweedie = model(x_train; device=_device)
mean((pred_train_tweedie .- y_train) .^ 2)

# MAE
config = EvoTreeRegressor(;
    loss=:mae,
    metric=:mse,
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=1.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
pred_train_mae = model(x_train; device=_device)

# using XGBoost
# params_xgb = Dict(
#     :num_round => 1,
#     :max_depth => 8,
#     :eta => 1.0,
#     :objective => "reg:logistic",
#     :print_every_n => 5,
#     :subsample => 0.5,
#     :colsample_bytree => 0.5,
#     :tree_method => "hist", # hist/gpu_hist
#     :max_bin => 32,
# )
# dtrain = DMatrix(x_train, y_train)
# watchlist = Dict("train" => DMatrix(x_train, y_train))
# m_xgb = xgboost(dtrain; watchlist, verbosity=0, eval_metric="mae", params_xgb...)
# pred_train_xgb = XGBoost.predict(m_xgb, x_train)

###########################################
# plot
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
    color="navy",
    linewidth=1,
    label="mse",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_logistic[x_perm],
    color="darkred",
    linewidth=1,
    label="logloss",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_poisson[x_perm],
    color="green",
    linewidth=1,
    label="poisson",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_gamma[x_perm],
    color="pink",
    linewidth=1,
    label="gamma",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_tweedie[x_perm],
    color="orange",
    linewidth=1,
    label="tweedie",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_mae[x_perm],
    color="lightblue",
    linewidth=1,
    label="mae",
)
# lines!(ax,
#     x_train[x_perm, 1],
#     pred_train_xgb[x_perm],
#     color="black",
#     linewidth=1,
#     label="xgb-logistic",
# )
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("docs/src/assets/bagging-regression-sinus-$tree_type-$_device.png", f)

###############################
## gaussian
###############################
config = EvoTreeGaussian(;
    nrounds=1,
    bagging_size=16,
    early_stopping_rounds=50,
    nbins=32,
    L2=0.0,
    # gamma=0.1,
    eta=1.0,
    max_depth=9,
    min_weight=8,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
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
pred_train_gaussian = model(x_train; device=_device)

pred_gauss = [
    Distributions.Normal(pred_train_gaussian[i, 1], pred_train_gaussian[i, 2]) for
    i in axes(pred_train_gaussian, 1)
]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

mean(y_train .< pred_q80)
mean(y_train .< pred_q20)

###########################################
# plot
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
    pred_train_gaussian[x_perm, 1],
    color="navy",
    linewidth=1,
    label="mu",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_gaussian[x_perm, 2],
    color="darkred",
    linewidth=1,
    label="sigma",
)
# lines!(ax,
#     x_train[x_perm, 1],
#     pred_q20[x_perm, 1],
#     color="green",
#     linewidth=1,
#     label="q20",
# )
# lines!(ax,
#     x_train[x_perm, 1],
#     pred_q80[x_perm, 1],
#     color="green",
#     linewidth=1,
#     label="q80",
# )
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("docs/src/assets/bagging-gaussian-sinus-$tree_type-$_device.png", f)

###############################
## Quantiles
###############################
# q50
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.5,
    nrounds=1,
    bagging_size=16,
    nbins=32,
    eta=1.0,
    L2=0.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
# 116.822 ms (74496 allocations: 36.41 MiB) for 100 iterations
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_q50 = model(x_train)
@info sum(pred_train_q50 .< y_train) / length(y_train)

# q20
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.2,
    nrounds=1,
    bagging_size=16,
    nbins=32,
    eta=1.0,
    L2=0.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
@time pred_train_q20 = model(x_train)
@info sum(pred_train_q20 .> y_train) / length(y_train)

# q80
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.8,
    nrounds=1,
    bagging_size=1,
    nbins=32,
    L2=0.0,
    eta=1.0,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
@time pred_train_q80 = model(x_train)
@info sum(pred_train_q80 .> y_train) / length(y_train)

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
    pred_train_q50[x_perm],
    color="navy",
    linewidth=1,
    label="Median",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_q20[x_perm],
    color="darkred",
    linewidth=1,
    label="Q20",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_q80[x_perm],
    color="darkgreen",
    linewidth=1,
    label="Q80",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("docs/src/assets/bagging-quantiles-sinus-$tree_type-$_device.png", f)


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
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_cred_var = model(x_train; device=_device)
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
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_cred_std = model(x_train; device=_device)
sqrt(mean((pred_train_cred_std .- y_train) .^ 2))

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
save("docs/src/assets/bagging-credibility-sinus-$tree_type-$_device.png", f)
