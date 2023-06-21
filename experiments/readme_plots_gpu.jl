using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using Plots
using Revise
using EvoTrees
using EvoTrees: predict, sigmoid, logit
# using ProfileView

# prepare a dataset
device = "gpu"
Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

# linear
params1 = EvoTreeRegressor(
    loss=:linear,
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

# @time model = fit_evotree(params1; x_train, y_train);
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    metric=:mse,
    print_every_n=25,
    early_stopping_rounds=50,
    device
);
# model, logger = fit_evotree(params1; x_train, y_train, metric=:mse, x_eval, y_eval, early_stopping_rounds=20, print_every_n=10, return_logger=true);
@time pred_train_linear_cpu = model(x_train)
@time pred_train_linear_gpu = model(x_train; device)
sum(pred_train_linear_gpu .- pred_train_linear_cpu)

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = predict(model, x_train)
mean(abs.(pred_train_linear .- y_train))
sqrt(mean((pred_train_linear .- y_train) .^ 2))

# logistic / cross-entropy
params1 = EvoTreeRegressor(
    T=Float32,
    loss=:logistic,
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
    metric=:logloss,
    print_every_n=25,
    early_stopping_rounds=50,
    device
);
@time pred_train_logistic = model(x_train; device)
sqrt(mean((pred_train_logistic .- y_train) .^ 2))

# poisson
params1 = EvoTreeCount(
    T=Float32,
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
    metric=:poisson,
    print_every_n=25,
    early_stopping_rounds=50,
    device
);
@time pred_train_poisson = model(x_train; device)
sqrt(mean((pred_train_poisson .- y_train) .^ 2))

# gamma
params1 = EvoTreeRegressor(
    T=Float32,
    loss=:gamma,
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.03,
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
    metric=:gamma,
    print_every_n=25,
    early_stopping_rounds=50,
    device
);
@time pred_train_gamma = model(x_train; device)
sqrt(mean((pred_train_gamma .- y_train) .^ 2))


# tweedie
params1 = EvoTreeRegressor(
    loss=:tweedie,
    nrounds=500,
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
    metric=:tweedie,
    print_every_n=25,
    early_stopping_rounds=50,
    device
);
@time pred_train_tweedie = model(x_train; device)
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
    pred_train_linear[x_perm],
    color="navy",
    linewidth=1.5,
    label="Linear",
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
    pred_train_poisson[x_perm],
    color="green",
    linewidth=1.5,
    label="Poisson",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_gamma[x_perm],
    color="pink",
    linewidth=1.5,
    label="Gamma",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_tweedie[x_perm],
    color="orange",
    linewidth=1.5,
    label="Tweedie",
)
savefig("figures/regression_sinus_gpu.png")


###############################
## gaussian
###############################
params1 = EvoTreeGaussian(
    nrounds=500,
    nbins=64,
    lambda=0.1,
    gamma=0.1,
    eta=0.05,
    max_depth=6,
    min_weight=20,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
)

@time model = fit_evotree(params1; x_train, y_train);
@time model = fit_evotree(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
    early_stopping_rounds=50,
    metric=:gaussian,
    device
);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train_gaussian = model(x_train; device)

pred_gauss = [
    Distributions.Normal(pred_train_gaussian[i, 1], pred_train_gaussian[i, 2]) for
    i in axes(pred_train_gaussian, 1)
]
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
    pred_train_gaussian[x_perm, 1],
    color="navy",
    linewidth=1.5,
    label="mu",
)
plot!(
    x_train[:, 1][x_perm],
    pred_train_gaussian[x_perm, 2],
    color="darkred",
    linewidth=1.5,
    label="sigma",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q20[x_perm, 1],
    color="green",
    linewidth=1.5,
    label="q20",
)
plot!(
    x_train[:, 1][x_perm],
    pred_q80[x_perm, 1],
    color="green",
    linewidth=1.5,
    label="q80",
)
savefig("figures/gaussian-sinus-gpu.png")
