using BenchmarkTools
using Statistics
using StatsBase: sample
using Distributions
using Random
using CUDA
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit
using CairoMakie

# prepare a dataset
tree_type = :binary # binary/oblivious
_device = :cpu
assets = joinpath(@__DIR__, "..", "docs", "src", "assets")

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
x = x_train[:, 1]

train_kw = (
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device,
)

function fit_and_predict(config)
    @time model = fit(config; x_train, y_train, x_eval, y_eval, print_every_n=25)
    @time pred = model(x_train; device=_device)
    return pred
end

function plot_sinus(x, y, series; name)
    perm = sortperm(x)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
    scatter!(ax, x[perm], y[perm]; color="#BBB", markersize=2)
    for s in series
        lines!(ax, x[perm], s.y[perm]; color=s.color, linewidth=1, label=s.label)
    end
    Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
    save(joinpath(assets, "$name-$tree_type-$_device.svg"), f)
    return f
end

###############################
# Point-prediction losses
###############################
point_specs = [
    (ctor=EvoTreeRegressor, extra=(loss=:mse, L2=0.0), color="navy", label="mse"),
    (ctor=EvoTreeRegressor, extra=(loss=:logloss, L2=1.0), color="darkred", label="logloss"),
    (ctor=EvoTreeCount, extra=(L2=1.0,), color="green", label="poisson"),
    (ctor=EvoTreeRegressor, extra=(loss=:gamma, L2=1.0), color="pink", label="gamma"),
    (ctor=EvoTreeRegressor, extra=(loss=:tweedie, L2=1.0), color="orange", label="tweedie"),
    (ctor=EvoTreeRegressor, extra=(loss=:mae, L2=0.0), color="lightblue", label="mae"),
]

point_series = map(point_specs) do spec
    pred = fit_and_predict(spec.ctor(; train_kw..., spec.extra...))
    @info spec.label rmse = sqrt(mean((pred .- y_train) .^ 2))
    (y=pred, color=spec.color, label=spec.label)
end
plot_sinus(x, y_train, point_series; name="regression-sinus")

###############################
# MLE: Gaussian / Logistic
###############################
mle_specs = [
    (loss=:gaussian_mle, Dist=Normal, scale="sigma", name="gaussian-sinus"),
    (loss=:logistic_mle, Dist=Logistic, scale="scale", name="logistic-sinus"),
]

for spec in mle_specs
    pred = fit_and_predict(EvoTreeMLE(; train_kw..., loss=spec.loss, L2=0.0, min_weight=8, seed=123))
    dists = [spec.Dist(pred[i, 1], pred[i, 2]) for i in axes(pred, 1)]
    q20 = quantile.(dists, 0.2)
    q80 = quantile.(dists, 0.8)
    @info spec.loss coverage_q20 = mean(y_train .< q20) coverage_q80 = mean(y_train .< q80)
    plot_sinus(x, y_train, [
        (y=pred[:, 1], color="navy", label="mu"),
        (y=pred[:, 2], color="darkred", label=spec.scale),
        (y=q20, color="green", label="q20"),
        (y=q80, color="green", label="q80"),
    ]; name=spec.name)
end

###############################
# Quantiles
###############################
quantile_specs = [
    (alpha=0.5, eta=0.1, color="navy", label="Median"),
    (alpha=0.2, eta=0.1, color="darkred", label="Q20"),
    (alpha=0.8, eta=0.2, color="darkgreen", label="Q80"),
]

quantile_series = map(quantile_specs) do spec
    pred = fit_and_predict(EvoTreeRegressor(;
        train_kw...,
        loss=:quantile,
        alpha=spec.alpha,
        eta=spec.eta,
        L2=1.0,
    ))
    @info spec.label coverage = mean(y_train .<= pred)
    (y=pred, color=spec.color, label=spec.label)
end
plot_sinus(x, y_train, quantile_series; name="quantiles-sinus")

###############################
# MultiQuantile
###############################
pred_q = fit_and_predict(EvoTreeRegressor(;
    train_kw...,
    loss=:multiquantile,
    alphas=[0.2, 0.5, 0.8],
    L2=1.0,
))
@info [mean(p .> y_train) for p in eachcol(pred_q)]
plot_sinus(x, y_train, [
    (y=pred_q[:, 1], color="darkred", label="Q20"),
    (y=pred_q[:, 2], color="navy", label="Median"),
    (y=pred_q[:, 3], color="darkgreen", label="Q80"),
]; name="multiquantile-sinus")

###############################
# Credibility losses
###############################
cred_specs = [
    (loss=:cred_var, color="navy", label="cred_var"),
    (loss=:cred_std, color="darkred", label="cred_std"),
]

cred_series = map(cred_specs) do spec
    pred = fit_and_predict(EvoTreeRegressor(; train_kw..., loss=spec.loss, L2=1.0, lambda=1.0))
    @info spec.label rmse = sqrt(mean((pred .- y_train) .^ 2))
    (y=pred, color=spec.color, label=spec.label)
end
plot_sinus(x, y_train, cred_series; name="credibility-sinus")
