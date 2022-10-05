using Revise
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
    loss=:linear, metric=:mae,
    nrounds=20, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=1.0,
    rng=123)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=5);
@time pred_train_linear = predict(model, x_train);
mean(abs.(pred_train_linear .- y_train))

# offset
@time model_offset = fit_evotree(params1; x_train, y_train, x_eval, y_eval, offset_train=pred_train_linear, print_every_n=25);
@time pred_train_linear_offset = pred_train_linear .+ predict(model_offset, x_train);
mean(abs.(pred_train_linear_offset .- y_train))

x_perm = sortperm(x_train[:, 1])
plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train_linear[x_perm], color="navy", linewidth=1.5, label="Linear")
plot!(x_train[:, 1][x_perm], pred_train_linear_offset[x_perm], color="lightblue", linewidth=1.5, label="Linear-Offset")

###############################
## gaussian - cpu
###############################
params1 = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=20, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, rng=123)

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=5);
@time pred_train = EvoTrees.predict(model, x_train);
@time pred_eval = EvoTrees.predict(model, x_eval);

@time model_res = fit_evotree(params1; x_train, y_train, offset_train=copy(pred_train), x_eval, y_eval, offset_eval=copy(pred_eval), print_every_n=5);
@time pred_train_res = EvoTrees.predict(model_res, x_train);
pred_train_stack = copy(pred_train)
pred_train_stack[:, 2] .= log.(pred_train_stack[:, 2])
pred_train_res_stack = copy(pred_train_res)
pred_train_res_stack[:, 2] .= log.(pred_train_res_stack[:, 2])
pred_train_tot = pred_train_stack + pred_train_res_stack
pred_train_tot[:, 2] .= exp.(pred_train_tot[:, 2])

x_perm = sortperm(x_train[:, 1])
plot(x_train[:, 1], y_train, ms=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 1], color="navy", linewidth=1.5, label="mu")
plot!(x_train[:, 1][x_perm], pred_train_res[x_perm, 1], color="blue", linewidth=1.5, label="mu-res")
plot!(x_train[:, 1][x_perm], pred_train_tot[x_perm, 1], color="lightblue", linewidth=1.5, label="mu-tot")

plot!(x_train[:, 1][x_perm], pred_train[x_perm, 2], color="darkred", linewidth=1.5, label="sigma")
plot!(x_train[:, 1][x_perm], pred_train_res[x_perm, 2], color="red", linewidth=1.5, label="sigma-res")
plot!(x_train[:, 1][x_perm], pred_train_tot[x_perm, 2], color="pink", linewidth=1.5, label="sigma-tot")


###############################
## gaussian - gpu
###############################
params1 = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=20, nbins=64,
    lambda=0.1, gamma=0.1, eta=0.05,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, rng=123,
    device="gpu")

@time model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=5);
@time pred_train = EvoTrees.predict(model, x_train);
@time pred_eval = EvoTrees.predict(model, x_eval);

@time model_res = fit_evotree(params1; x_train, y_train, offset_train=copy(pred_train), x_eval, y_eval, offset_eval=copy(pred_eval), print_every_n=5);
@time pred_train_res = EvoTrees.predict(model_res, x_train);
pred_train_stack = copy(pred_train)
pred_train_stack[:, 2] .= log.(pred_train_stack[:, 2])
pred_train_res_stack = copy(pred_train_res)
pred_train_res_stack[:, 2] .= log.(pred_train_res_stack[:, 2])
pred_train_tot = pred_train_stack + pred_train_res_stack
pred_train_tot[:, 2] .= exp.(pred_train_tot[:, 2])

x_perm = sortperm(x_train[:, 1])
plot(x_train[:, 1], y_train, ms=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
plot!(x_train[:, 1][x_perm], pred_train[x_perm, 1], color="darkblue", linewidth=1.5, label="mu")
plot!(x_train[:, 1][x_perm], pred_train_res[x_perm, 1], color="blue", linewidth=1.5, label="mu-res")
plot!(x_train[:, 1][x_perm], pred_train_tot[x_perm, 1], color="lightblue", linewidth=1.5, label="mu-tot")

plot!(x_train[:, 1][x_perm], pred_train[x_perm, 2], color="darkred", linewidth=1.5, label="sigma")
plot!(x_train[:, 1][x_perm], pred_train_res[x_perm, 2], color="red", linewidth=1.5, label="sigma-res")
plot!(x_train[:, 1][x_perm], pred_train_tot[x_perm, 2], color="pink", linewidth=1.5, label="sigma-tot")
