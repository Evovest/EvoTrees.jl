using Statistics
using StatsBase: sample
using EvoTrees: sigmoid, logit

# prepare a dataset
features = rand(10_000) .* 2
X = reshape(features, (size(features)[1], 1))
noise = exp.(randn(length(X)))
Y = 2 .+ 3 .* X .+ noise
W = noise
𝑖 = collect(1:size(X, 1))
seed = 123

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]
w_train = W[𝑖_train]
w_eval = W[𝑖_eval]

# linear - no weights
params1 = EvoTreeRegressor(T=Float32, device="cpu",
    loss=:linear, metric=:mse,
    nrounds=100, nbins=100,
    lambda=0.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
preds_no_weight = predict(model, X_train)

# linear - weighted
params1 = EvoTreeRegressor(T=Float32, device="cpu",
    loss=:linear, metric=:mse,
    nrounds=100, nbins=100,
    lambda=0.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model, cache = EvoTrees.init_evotree(params1, X_train, Y_train, W_train)
model = fit_evotree(params1; x_train, y_train, w_train, x_eval, y_eval, print_every_n=25);
preds_weighted_1 = predict(model, X_train)

params1 = EvoTreeRegressor(T=Float32, device="cpu",
    loss=:linear, metric=:mse,
    nrounds=100, nbins=100,
    lambda=0.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

model, cache = EvoTrees.init_evotree(params1, X_train, Y_train, W_train)
model = fit_evotree(params1; x_train, y_train, w_train, x_eval, y_eval, w_eval, print_every_n=25);
preds_weighted_2 = predict(model, X_train)

params1 = EvoTreeRegressor(T=Float32, device="cpu",
    loss=:linear, metric=:mse,
    nrounds=100, nbins=100,
    lambda=0.0, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=0.0,
    rowsample=0.5, colsample=1.0, rng=seed)

w_train_3 = ones(eltype(Y_train), size(Y_train)) .* 5

model, cache = EvoTrees.init_evotree(params1, X_train, Y_train, W_train_3)
model = fit_evotree(params1; x_train, y_train, w_train=w_train_3, x_eval, y_eval, print_every_n=25);
preds_weighted_3 = predict(model, X_train)

sum(abs.(preds_no_weight .- preds_weighted_3))
cor(preds_no_weight, preds_weighted_3)

# using Plots
# # using Colors
# x_perm = sortperm(X_train[:,1])
# plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
# plot!(X_train[:,1][x_perm], preds_no_weight[x_perm], color="navy", linewidth=1.5, label="No weights")
# plot!(X_train[:,1][x_perm], preds_weighted[x_perm], color="red", linewidth=1.5, label="Weighted")