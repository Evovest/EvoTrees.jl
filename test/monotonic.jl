using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using EvoTrees: sigmoid, logit

# prepare a dataset
features = rand(10_000) .* 2.5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y)) .* 0.2
Y = sigmoid(Y)
𝑖 = collect(1:size(X, 1))
seed = 123

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

######################################
### Linear - CPU
######################################
# benchmark
params1 = EvoTreeRegressor(
    device="cpu",
    loss=:linear, metric=:mse,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeRegressor(
    device="cpu",
    loss=:linear, metric=:mse,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.5,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


######################################
### Linear - GPU
######################################
# benchmark
params1 = EvoTreeRegressor(
    device="gpu",
    loss=:linear, metric=:mse,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeRegressor(
    device="gpu",
    loss=:linear, metric=:mse,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.5,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


######################################
### Logistic - CPU
######################################
# benchmark
params1 = EvoTreeRegressor(
    device="cpu",
    loss=:logistic, metric=:logloss,
    nrounds=200, nbins=32,
    lambda=0.05, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeRegressor(
    device="cpu",
    loss=:logistic, metric=:logloss,
    nrounds=200, nbins=32,
    lambda=0.05, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


######################################
### Logistic - GPU
######################################
# benchmark
params1 = EvoTreeRegressor(
    device="gpu",
    loss=:logistic, metric=:logloss,
    nrounds=200, nbins=32,
    lambda=0.05, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeRegressor(
    device="gpu",
    loss=:logistic, metric=:logloss,
    nrounds=200, nbins=32,
    lambda=0.05, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


######################################
### Gaussian - CPU
######################################
# linear - benchmark
params1 = EvoTreeGaussian(
    device="cpu",
    metric=:gaussian,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeGaussian(
    device="cpu",
    metric=:gaussian,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.5,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


######################################
### Gaussian - GPU
######################################
# linear - benchmark
params1 = EvoTreeGaussian(
    device="gpu",
    metric=:gaussian,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_ref = predict(model, X_train);

# monotonic constraint
params1 = EvoTreeGaussian(
    device="gpu",
    metric=:gaussian,
    nrounds=200, nbins=32,
    lambda=1.0, gamma=0.0, eta=0.5,
    max_depth=6, min_weight=0.0,
    monotone_constraints=Dict(1 => 1),
    rowsample=0.5, colsample=1.0, rng=seed)

model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25);
preds_mono = predict(model, X_train);

using Plots
using Colors
x_perm = sortperm(X_train[:, 1])
plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")
