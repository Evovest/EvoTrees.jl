using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using DataFrames
using Random
using CairoMakie
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

obs = [-6, -1, 1]
preds = -8:0.1:3

###########################################
# mse
###########################################
mse(p, y) = (p - y)^2
mse_loss = [mse.(x, preds) for x in obs]
push!(mse_loss, reduce(+, mse_loss))

x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(
    f[1, 1],
    xticks=-8:2:3,       # show 1,3,5,7,9
    xlabel="target",
    xlabelsize=18,
    ylabel="loss",
    ylabelsize=18,
    xticklabelsize=18,
    yticklabelsize=18,
    title="MSE",
    titlesize=20
)
scatter!(
    ax,
    obs,
    zero(obs),
    # label="raw",
    markersize=20,
    color="#023572"
)
for i in 1:length(obs)
    lines!(
        ax,
        preds,
        mse_loss[i],
        linewidth=1,
        color="#5891d5"
    )
end
lines!(
    ax,
    preds,
    mse_loss[4],
    linewidth=3,
    color="#26a671"
)
f
save(joinpath(@__DIR__, "mse-loss.png"), f)

###########################################
# mse
###########################################
mae(p, y) = abs(p - y)
mae_loss = [mae.(x, preds) for x in obs]
push!(mae_loss, reduce(+, mae_loss))

x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(
    f[1, 1],
    xticks=-8:2:3,       # show 1,3,5,7,9
    xlabel="target",
    xlabelsize=18,
    ylabel="loss",
    ylabelsize=18,
    xticklabelsize=18,
    yticklabelsize=18,
    title="MAE",
    titlesize=20
)
scatter!(
    ax,
    obs,
    zero(obs),
    # label="raw",
    markersize=20,
    color="#023572"
)
for i in 1:length(obs)
    lines!(
        ax,
        preds,
        mae_loss[i],
        linewidth=1,
        color="#5891d5"
    )
end
lines!(
    ax,
    preds,
    mae_loss[4],
    linewidth=3,
    color="#e5616c"
)
f
save(joinpath(@__DIR__, "mae-loss.png"), f)
