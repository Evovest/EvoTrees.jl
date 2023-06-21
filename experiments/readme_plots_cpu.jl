using Revise
using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using Plots
using EvoTrees
using EvoTrees: predict, sigmoid, logit
# using ProfileView

# prepare a dataset
Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
is = sample(is, length(is), replace=false)
train_size = 0.8
i_train = is[1:floor(Int, train_size * size(is, 1))]
i_eval = is[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

# linear
params1 = EvoTreeRegressor(
    loss=:mse,
    nrounds=200,
    nbins=64,
    lambda=0.01,
    gamma=0.1,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=122,
)

@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    metric=:mse,
    print_every_n=25,
    early_stopping_rounds=20
);
# laptop: 51.651 ms (237548 allocations: 23.94 MiB)
# @btime model = fit_evotree(params1; x_train, y_train, x_eval = x_eval, y_eval = y_eval, metric = :mse, print_every_n = 999, verbosity=0);
# Profile.clear()  # in case we have any previous profiling data
# @profile fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# ProfileView.view()
model, logger = fit_evotree(
    params1;
    x_train,
    y_train,
    metric=:mse,
    x_eval,
    y_eval,
    early_stopping_rounds=20,
    print_every_n=10,
    return_logger=true
);
plot(logger[:metrics])

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = model(x_train);
@time pred_eval_linear = model(x_eval)
mean((pred_train_linear .- y_train) .^ 2)
mean((pred_eval_linear .- y_eval) .^ 2)

# linear weighted
params1 = EvoTreeRegressor(
    T=Float64,
    loss=:linear,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
)

# W_train = ones(eltype(Y_train), size(Y_train)) .* 5
w_train = rand(eltype(y_train), size(y_train)) .+ 0
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    w_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:mse
);
# 67.159 ms (77252 allocations: 28.06 MiB)
# @time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 999);
# @btime model = fit_evotree($params1, $X_train, $Y_train, X_eval = $X_eval, Y_eval = $Y_eval);
# Profile.clear()  # in case we have any previous profiling data
# @profile fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# ProfileView.view()

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear_w = model(x_train);
@time pred_eval_linear_w = model(x_eval)
mean(abs.(pred_train_linear_w .- y_train))
sqrt(mean((pred_train_linear_w .- y_train) .^ 2))

# logistic / cross-entropy
params1 = EvoTreeRegressor(
    loss=:logistic,
    nrounds=200,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)

@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:logloss
);
# 218.040 ms (123372 allocations: 34.71 MiB)
# @btime model = fit_evotree($params1, $X_train, $Y_train, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_logistic = model(x_train);
@time pred_eval_logistic = model(x_eval)
sqrt(mean((pred_train_logistic .- y_train) .^ 2))

# L1
params1 = EvoTreeRegressor(
    loss=:l1,
    alpha=0.5,
    nrounds=500,
    nbins=64,
    lambda=0.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:mae
);
@time pred_train_L1 = model(x_train)
@time pred_eval_L1 = model(x_eval)
sqrt(mean((pred_train_L1 .- y_train) .^ 2))

x_perm = sortperm(x_train[:, 1])
plot(
    x_train,
    y_train,
    msize=0.5,
    mcolor="darkgray",
    mswidth=0,
    background_color=RGB(1, 1, 1),
    seriestype=:scatter,
    xaxis=("feature"),
    yaxis=("target"),
    legend=true,
    label="",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_linear[x_perm],
    color="navy",
    linewidth=1.5,
    label="Linear",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_linear_w[x_perm],
    color="lightblue",
    linewidth=1.5,
    label="LinearW",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_logistic[x_perm],
    color="darkred",
    linewidth=1.5,
    label="Logistic",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_L1[x_perm],
    color="darkgreen",
    linewidth=1.5,
    label="L1",
)
savefig("figures/regression_sinus.png")

# Poisson
params1 = EvoTreeCount(
    loss=:poisson,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:poisson
);
@time pred_train_poisson = model(x_train);
sqrt(mean((pred_train_poisson .- y_train) .^ 2))

# Gamma
params1 = EvoTreeRegressor(
    loss=:gamma,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.02,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:gamma
);
@time pred_train_gamma = model(x_train);
sqrt(mean((pred_train_gamma .- y_train) .^ 2))

# Tweedie
params1 = EvoTreeRegressor(
    loss=:tweedie,
    nrounds=500,
    nbins=64,
    lambda=0.5,
    gamma=0.1,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:tweedie
);
@time pred_train_tweedie = model(x_train);
sqrt(mean((pred_train_tweedie .- y_train) .^ 2))

x_perm = sortperm(x_train[:, 1])
plot(
    x_train,
    y_train,
    msize=0.5,
    mcolor="darkgray",
    mswidth=0,
    background_color=RGB(1, 1, 1),
    seriestype=:scatter,
    xaxis=("feature"),
    yaxis=("target"),
    legend=true,
    label="",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_poisson[x_perm],
    color="navy",
    linewidth=1.5,
    label="Poisson",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_gamma[x_perm],
    color="lightblue",
    linewidth=1.5,
    label="Gamma",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_tweedie[x_perm],
    color="darkred",
    linewidth=1.5,
    label="Tweedie",
)
savefig("figures/regression_sinus2.png")


###############################
## Quantiles
###############################
# q50
params1 = EvoTreeRegressor(
    loss=:quantile,
    alpha=0.5,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:mae
);
# 116.822 ms (74496 allocations: 36.41 MiB) for 100 iterations
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_q50 = model(x_train)
sum(pred_train_q50 .< y_train) / length(y_train)

# q20
params1 = EvoTreeRegressor(
    loss=:quantile,
    alpha=0.2,
    nrounds=300,
    nbins=64,
    lambda=0.1,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
@time pred_train_q20 = model(x_train)
sum(pred_train_q20 .< y_train) / length(y_train)

# q80
params1 = EvoTreeRegressor(
    loss=:quantile,
    alpha=0.8,
    nrounds=300,
    nbins=64,
    lambda=0.1,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
@time pred_train_q80 = model(x_train)
sum(pred_train_q80 .< y_train) / length(y_train)

x_perm = sortperm(x_train[:, 1])
plot(
    x_train,
    y_train,
    ms=0.5,
    mcolor="darkgray",
    mswidth=0,
    background_color=RGB(1, 1, 1),
    seriestype=:scatter,
    xaxis=("feature"),
    yaxis=("target"),
    legend=true,
    label="",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_q50[x_perm],
    color="navy",
    linewidth=1.5,
    label="Median",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_q20[x_perm],
    color="darkred",
    linewidth=1.5,
    label="Q20",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_q80[x_perm],
    color="darkgreen",
    linewidth=1.5,
    label="Q80",
)
savefig("figures/quantiles_sinus.png")


###############################
## gaussian
###############################
params1 = EvoTreeMLE(
    T=Float64,
    loss=:gaussian,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.05,
    max_depth=6,
    min_weight=10.0,
    rowsample=1.0,
    colsample=1.0,
    rng=123,
)

@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:gaussian
);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train = model(x_train);
# @btime pred_train = EvoTrees.predict(model, X_train);

pred_gauss =
    [Distributions.Normal(pred_train[i, 1], pred_train[i, 2]) for i in axes(pred_train, 1)]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

mean(y_train .< pred_q80)
mean(y_train .< pred_q20)

x_perm = sortperm(x_train[:, 1])
plot(
    x_train[:, 1],
    y_train,
    ms=0.5,
    mcolor="darkgray",
    mswidth=0,
    background_color=RGB(1, 1, 1),
    seriestype=:scatter,
    xaxis=("feature"),
    yaxis=("target"),
    legend=true,
    label="",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train[x_perm, 1],
    color="navy",
    linewidth=1.5,
    label="mu",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train[x_perm, 2],
    color="darkred",
    linewidth=1.5,
    label="sigma",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q20[x_perm, 1],
    color="darkgreen",
    linewidth=1.5,
    label="q20",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q80[x_perm, 1],
    color="darkgreen",
    linewidth=1.5,
    label="q80",
)
savefig("figures/gaussian-sinus.png")


###############################
## Logistic
###############################
params1 = EvoTrees.EvoTreeMLE(
    loss=:logistic,
    nrounds=500,
    nbins=64,
    lambda=1.0,
    gamma=0.1,
    eta=0.03,
    max_depth=6,
    min_weight=1.0,
    rowsample=1.0,
    colsample=1.0,
    rng=123,
)

@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:logistic_mle
);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train = model(x_train);
# @btime pred_train = EvoTrees.predict(model, X_train);

pred_logistic = [
    Distributions.Logistic(pred_train[i, 1], pred_train[i, 2]) for i in axes(pred_train, 1)
]
pred_q80 = quantile.(pred_logistic, 0.8)
pred_q20 = quantile.(pred_logistic, 0.2)

mean(y_train .< pred_q80)
mean(y_train .< pred_q20)

x_perm = sortperm(x_train[:, 1])
plot(
    x_train[:, 1],
    y_train,
    ms=0.5,
    mcolor="darkgray",
    mswidth=0,
    background_color=RGB(1, 1, 1),
    seriestype=:scatter,
    xaxis=("feature"),
    yaxis=("target"),
    legend=true,
    label="",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train[x_perm, 1],
    color="navy",
    linewidth=1.5,
    label="mu",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train[x_perm, 2],
    color="darkred",
    linewidth=1.5,
    label="s",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q20[x_perm, 1],
    color="darkgreen",
    linewidth=1.5,
    label="q20",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q80[x_perm, 1],
    color="darkgreen",
    linewidth=1.5,
    label="q80",
)
savefig("figures/logistic-sinus.png")
