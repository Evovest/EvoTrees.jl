using BenchmarkTools
using Statistics
using StatsBase: sample
using Random
using CairoMakie
using DataFrames
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit
using CUDA

device = :cpu
tree_type = :binary
assets = joinpath(@__DIR__, "..", "docs", "src", "assets")

# prepare a dataset
nobs = 10_000
Random.seed!(123)
x_num = rand(nobs) .* 5

function noisy_sin(x)
    y = sin.(x) .* 0.5 .+ 0.5
    y = logit(y) + randn(length(x)) .* 0.5
    return sigmoid(y)
end

dtot = DataFrame(x_num=x_num, y=noisy_sin(x_num), y2=noisy_sin(2x_num .- 2))
is = sample(1:nobs, nobs, replace=false)
ntrain = floor(Int, 0.8 * nobs)
dtrain = dtot[is[1:ntrain], :]
deval = dtot[is[ntrain+1:end], :]

train_kw = (
    nrounds=200,
    nbins=64,
    L2=0.1,
    gamma=0.05,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    seed=123,
    tree_type,
    device,
)

function fit_and_predict(config)
    @time model = fit(
        config,
        dtrain;
        feature_names=["x_num"],
        target_name=["y", "y2"],
        deval,
        print_every_n=25,
        verbosity=0,
    )
    @time pred = model(dtrain; device)
    return pred
end

# One color per target. `nparams=1` is a point prediction; `nparams=2` is MLE (μ, scale).
function plot_multi(pred; name, nparams=1)
    x = dtrain.x_num
    perm = sortperm(x)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
    for t in (
        (y=dtrain.y, color="#26a671", label="y", offset=0),
        (y=dtrain.y2, color="#e5616c", label="y2", offset=nparams),
    )
        scatter!(ax, x[perm], t.y[perm]; markersize=2, color=t.color, label=t.label)
        for k in 1:nparams
            lines!(ax, x[perm], pred[perm, t.offset+k]; linewidth=2, color=t.color)
        end
    end
    Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
    save(joinpath(assets, "multi-target-$name-$tree_type-$device.png"), f)
    return f
end

###############################
# Point-prediction losses
###############################
for loss in [:mse, :logloss, :gamma, :poisson, :tweedie, :mae, :quantile, :cred_std, :cred_var]
    pred = fit_and_predict(EvoTreeRegressor(; train_kw..., loss))
    plot_multi(pred; name=loss)
end

###############################
# MLE: Gaussian / Logistic
###############################
for loss in [:gaussian_mle, :logistic_mle]
    pred = fit_and_predict(EvoTreeMLE(; train_kw..., loss))
    plot_multi(pred; name=loss, nparams=2)
end
