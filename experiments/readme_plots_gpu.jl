using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using Plots
using Revise
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

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:mse,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.1, colsample=1.0,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
model_cpu = convert(EvoTrees.GBTree, model);
pred_train_linear_gpu = predict(model, X_train)
pred_train_linear_cpu = predict(model_cpu, X_train)
sum(pred_train_linear_gpu .- pred_train_linear_cpu)

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = predict(model, X_train)
mean(abs.(pred_train_linear .- Y_train))
sqrt(mean((pred_train_linear .- Y_train) .^ 2))

# logistic / cross-entropy
params1 = EvoTreeRegressor(T=Float32,
    loss=:logistic, metric=:logloss,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
@time pred_train_logistic = predict(model, X_train)
sqrt(mean((pred_train_logistic .- Y_train) .^ 2))

# poisson
params1 = EvoTreeCount(T=Float32,
    loss=:poisson, metric=:poisson,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
@time pred_train_poisson = predict(model, X_train)
sqrt(mean((pred_train_poisson .- Y_train) .^ 2))

# gamma
params1 = EvoTreeRegressor(T=Float32,
    loss=:gamma, metric=:gamma,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
@time pred_train_gamma = predict(model, X_train)
sqrt(mean((pred_train_gamma .- Y_train) .^ 2))


# tweedie
params1 = EvoTreeRegressor(T=Float32,
    loss=:tweedie, metric=:tweedie,
    nrounds=200, nbins=64,
    lambda=0.5, gamma=0.1, eta=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
@time pred_train_tweedie = predict(model, X_train)
sqrt(mean((pred_train_tweedie .- Y_train) .^ 2))

x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], pred_train_linear[x_perm], color="navy", linewidth=1.5, label="Linear")
plot!(X_train[:, 1][x_perm], pred_train_logistic[x_perm], color="darkred", linewidth=1.5, label="Logistic")
plot!(X_train[:,1][x_perm], pred_train_poisson[x_perm], color = "green", linewidth = 1.5, label = "Poisson")
plot!(X_train[:,1][x_perm], pred_train_gamma[x_perm], color = "pink", linewidth = 1.5, label = "Gamma")
plot!(X_train[:,1][x_perm], pred_train_tweedie[x_perm], color = "orange", linewidth = 1.5, label = "Tweedie")
savefig("figures/regression_sinus_gpu.png")


###############################
## gaussian
###############################
EvoTrees.CUDA.allowscalar(false)
params1 = EvoTreeGaussian(T=Float32,
    loss=:gaussian, metric=:gaussian,
    nrounds=200, nbins=64,
    lambda=1.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=5,
    rowsample=0.5, colsample=1.0, rng=123,
    device="gpu")

@time model = fit_evotree(params1, X_train, Y_train, print_every_n=25);
# @time model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train_gaussian = EvoTrees.predict(model, X_train)

pred_gauss = [Distributions.Normal(pred_train_gaussian[i, 1], pred_train_gaussian[i, 2]) for i in axes(pred_train_gaussian, 1)]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

mean(Y_train .< pred_q80)
mean(Y_train .< pred_q20)

x_perm = sortperm(X_train[:, 1])
plot(X_train[:, 1], Y_train, ms=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], pred_train_gaussian[x_perm, 1], color="navy", linewidth=1.5, label="mu")
plot!(X_train[:, 1][x_perm], pred_train_gaussian[x_perm, 2], color="darkred", linewidth=1.5, label="sigma")
plot!(X_train[:, 1][x_perm], pred_q20[x_perm, 1], color="green", linewidth=1.5, label="q20")
plot!(X_train[:, 1][x_perm], pred_q80[x_perm, 1], color="green", linewidth=1.5, label="q80")
savefig("figures/gaussian_sinus_gpu.png")