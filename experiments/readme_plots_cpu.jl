using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using Plots
using EvoTrees
using EvoTrees: sigmoid, logit
# using ProfileView

# prepare a dataset
Random.seed!(12)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(T=Float64,
    loss=:linear, metric=:mse,
    nrounds=200, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    rng=123)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
# 67.159 ms (77252 allocations: 28.06 MiB)
# @time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 999);
# @btime model = fit_evotree($params1, $X_train, $Y_train, X_eval = $X_eval, Y_eval = $Y_eval);
# Profile.clear()  # in case we have any previous profiling data
# @profile fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# ProfileView.view()

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = predict(model, x_train);
@time pred_eval_linear = predict(model, x_eval)
mean(abs.(pred_train_linear .- y_train))
sqrt(mean((pred_train_linear .- y_train) .^ 2))

# linear weighted
params1 = EvoTreeRegressor(T=Float64,
    loss=:linear,
    nrounds=200, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    rng=123)

# W_train = ones(eltype(Y_train), size(Y_train)) .* 5
w_train = rand(eltype(y_train), size(y_train)) .+ 0

@time model = fit_evotree(params1; x_train, y_train, w_train, x_eval, y_eval, print_every_n=25, metric=:mse);
# 67.159 ms (77252 allocations: 28.06 MiB)
# @time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 999);
# @btime model = fit_evotree($params1, $X_train, $Y_train, X_eval = $X_eval, Y_eval = $Y_eval);
# Profile.clear()  # in case we have any previous profiling data
# @profile fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# ProfileView.view()

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear_w = predict(model, x_train);
@time pred_eval_linear_w = predict(model, x_eval)
mean(abs.(pred_train_linear_w .- y_train))
sqrt(mean((pred_train_linear_w .- y_train) .^ 2))

# logistic / cross-entropy
params1 = EvoTreeRegressor(
    loss=:logistic,
    nrounds=200, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25, metric=:logloss);
# 218.040 ms (123372 allocations: 34.71 MiB)
# @btime model = fit_evotree($params1, $X_train, $Y_train, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_logistic = predict(model, x_train);
@time pred_eval_logistic = predict(model, x_eval)
sqrt(mean((pred_train_logistic .- y_train) .^ 2))

# L1
params1 = EvoTreeRegressor(
    loss=:L1, alpha=0.5,
    nrounds=200, nbins=64,
    lambda=0.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25, metric=:mae);
@time pred_train_L1 = predict(model, x_train)
@time pred_eval_L1 = predict(model, x_eval)
sqrt(mean((pred_train_L1 .- y_train) .^ 2))

x_perm = sortperm(x_train[:, 1])
plot(x_train, y_train, msize=0.5, mcolor="darkgray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train_linear[x_perm], color="navy", linewidth=1.5, label="Linear")
plot!(x_train[:, 1][x_perm], pred_train_linear_w[x_perm], color="lightblue", linewidth=1.5, label="LinearW")
plot!(x_train[:, 1][x_perm], pred_train_logistic[x_perm], color="darkred", linewidth=1.5, label="Logistic")
plot!(x_train[:, 1][x_perm], pred_train_L1[x_perm], color="darkgreen", linewidth=1.5, label="L1")
savefig("figures/regression_sinus.png")

# Poisson
params1 = EvoTreeCount(
    loss=:poisson,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25, metric=:poisson);
@time pred_train_poisson = predict(model, x_train);
sqrt(mean((pred_train_poisson .- y_train) .^ 2))

# Gamma
params1 = EvoTreeRegressor(
    loss=:gamma,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25, metric=:gamma);
@time pred_train_gamma = predict(model, x_train);
sqrt(mean((pred_train_gamma .- y_train) .^ 2))

# Tweedie
params1 = EvoTreeRegressor(
    loss=:tweedie,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25, metric=:tweedie);
@time pred_train_tweedie = predict(model, x_train);
sqrt(mean((pred_train_tweedie .- y_train) .^ 2))

x_perm = sortperm(x_train[:, 1])
plot(x_train, y_train, msize=0.5, mcolor="darkgray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train_poisson[x_perm], color="navy", linewidth=1.5, label="Poisson")
plot!(x_train[:, 1][x_perm], pred_train_gamma[x_perm], color="lightblue", linewidth=1.5, label="Gamma")
plot!(x_train[:, 1][x_perm], pred_train_tweedie[x_perm], color="darkred", linewidth=1.5, label="Tweedie")
savefig("figures/regression_sinus2.png")


###############################
## Quantiles
###############################
# q50
params1 = EvoTreeRegressor(
    loss=:quantile, alpha=0.5,
    nrounds=200, nbins=64,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
# 116.822 ms (74496 allocations: 36.41 MiB) for 100 iterations
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_q50 = predict(model, x_train)
sum(pred_train_q50 .< y_train) / length(y_train)

# q20
params1 = EvoTreeRegressor(
    loss=:quantile, alpha=0.2,
    nrounds=200, nbins=64,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
@time pred_train_q20 = predict(model, x_train)
sum(pred_train_q20 .< y_train) / length(y_train)

# q80
params1 = EvoTreeRegressor(
    loss=:quantile, alpha=0.8,
    nrounds=200, nbins=64,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0)
@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
@time pred_train_q80 = predict(model, x_train)
sum(pred_train_q80 .< y_train) / length(y_train)

x_perm = sortperm(x_train[:, 1])
plot(x_train, y_train, ms=0.5, mcolor="darkgray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train_q50[x_perm], color="navy", linewidth=1.5, label="Median")
plot!(x_train[:, 1][x_perm], pred_train_q20[x_perm], color="darkred", linewidth=1.5, label="Q20")
plot!(x_train[:, 1][x_perm], pred_train_q80[x_perm], color="darkgreen", linewidth=1.5, label="Q80")
savefig("figures/quantiles_sinus.png")


###############################
## gaussian
###############################
params1 = EvoTreeMLE(
    loss=:gaussian,
    nrounds=200, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, rng=123)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=10, metric=:gaussian);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train = EvoTrees.predict(model, x_train);
# @btime pred_train = EvoTrees.predict(model, X_train);

pred_gauss = [Distributions.Normal(pred_train[i, 1], pred_train[i, 2]) for i in axes(pred_train, 1)]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

mean(y_train .< pred_q80)
mean(y_train .< pred_q20)

x_perm = sortperm(x_train[:, 1])
plot(x_train[:, 1], y_train, ms=0.5, mcolor="darkgray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 1], color="navy", linewidth=1.5, label="mu")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 2], color="darkred", linewidth=1.5, label="sigma")
plot!(x_train[:, 1][x_perm], pred_q20[x_perm, 1], color="darkgreen", linewidth=1.5, label="q20")
plot!(x_train[:, 1][x_perm], pred_q80[x_perm, 1], color="darkgreen", linewidth=1.5, label="q80")
savefig("figures/gaussian-sinus.png")


###############################
## Logistic
###############################
params1 = EvoTrees.EvoTreeMLE(
    loss = :logistic,
    nrounds=200, nbins=64,
    lambda=1.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, rng=123)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=10, metric=:logistic);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train = EvoTrees.predict(model, x_train);
# @btime pred_train = EvoTrees.predict(model, X_train);

pred_logistic = [Distributions.Logistic(pred_train[i, 1], pred_train[i, 2]) for i in axes(pred_train, 1)]
pred_q80 = quantile.(pred_logistic, 0.8)
pred_q20 = quantile.(pred_logistic, 0.2)

mean(y_train .< pred_q80)
mean(y_train .< pred_q20)

x_perm = sortperm(x_train[:, 1])
plot(x_train[:, 1], y_train, ms=0.5, mcolor="darkgray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 1], color="navy", linewidth=1.5, label="mu")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 2], color="darkred", linewidth=1.5, label="s")
plot!(x_train[:, 1][x_perm], pred_q20[x_perm, 1], color="darkgreen", linewidth=1.5, label="q20")
plot!(x_train[:, 1][x_perm], pred_q80[x_perm, 1], color="darkgreen", linewidth=1.5, label="q80")
savefig("figures/logistic-sinus.png")