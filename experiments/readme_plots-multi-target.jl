using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using CairoMakie
using DataFrames
# using ProfileView
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit
using CUDA

device = :gpu
tree_type = :binary
path = joinpath(@__DIR__, "..", "docs", "src", "assets")

# prepare a dataset
nobs = 10_000
Random.seed!(123)
x_num = rand(nobs) .* 5

y = sin.(x_num) .* 0.5 .+ 0.5
y = logit(y) + randn(nobs) .* 0.5
y = sigmoid(y)
is = collect(1:nobs)
dtot = DataFrame(x_num=x_num, y=y)

y = sin.(2x_num .- 2) .* 0.5 .+ 0.5
y = logit(y) + randn(nobs) .* 0.5
y = sigmoid(y)
is = collect(1:nobs)
insertcols!(dtot, :y2 => y)

# train-eval split
is = sample(is, length(is), replace=false)
train_size = 0.8
i_train = is[1:floor(Int, train_size*size(is, 1))]
i_eval = is[(floor(Int, train_size*size(is, 1))+1):end]

dtrain = dtot[i_train, :]
deval = dtot[i_eval, :]

############################################
# regressions
############################################
loss_list = [:mse, :logloss, :gamma, :poisson, :tweedie, :mae, :quantile, :cred_std, :cred_var]
# loss_list = [:cred_std, :cred_var]

for loss in loss_list
    config = EvoTreeRegressor(;
        loss,
        tree_type,
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
        device,
    )

    @time model = fit(
        config,
        dtrain;
        feature_names=["x_num"],
        target_name=["y", "y2"],
        deval,
        print_every_n=25,
        verbosity=0
    );
    @time pred = model(dtrain; device);

    ###########################################
    # plot
    ###########################################
    x_perm = sortperm(dtrain.x_num)
    f = Figure()
    ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
    scatter!(ax,
        dtrain.x_num[x_perm],
        dtrain.y[x_perm],
        markersize=2,
        label="y",
        color="#26a671",
    )
    lines!(ax,
        dtrain.x_num[x_perm],
        pred[x_perm, 1],
        linewidth=2,
        # label="y",
        color="#26a671",
    )
    scatter!(ax,
        dtrain.x_num[x_perm],
        dtrain.y2[x_perm],
        markersize=2,
        label="y2",
        color="#e5616c",
    )
    lines!(ax,
        dtrain.x_num[x_perm],
        pred[x_perm, 2],
        linewidth=2,
        # label="y2",
        color="#e5616c",
    )
    Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
    f
    save("$path/multi-target-$loss-$tree_type-$device.svg", f)
end

###########################################
# gaussian-MLE
###########################################
config = EvoTreeMLE(;
    loss=:gaussian_mle,
    tree_type,
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
    device,
)
@time model = fit(
    config,
    dtrain;
    feature_names=["x_num"],
    target_name=["y", "y2"],
    # deval,
    print_every_n=25,
    verbosity=0
);
@time pred_gaussian = model(dtrain; device);

x_perm = sortperm(dtrain.x_num)
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    dtrain.x_num[x_perm],
    dtrain.y[x_perm],
    markersize=2,
    # label="y",
    color="#26a671",
)
lines!(ax,
    dtrain.x_num[x_perm],
    pred_gaussian[x_perm, 1],
    linewidth=2,
    label="y",
    color="#26a671",
)
lines!(ax,
    dtrain.x_num[x_perm],
    pred_gaussian[x_perm, 2],
    linewidth=2,
    # label="y",
    color="#26a671",
)
scatter!(ax,
    dtrain.x_num[x_perm],
    dtrain.y2[x_perm],
    markersize=2,
    # label="y2",
    color="#e5616c",
)
lines!(ax,
    dtrain.x_num[x_perm],
    pred_gaussian[x_perm, 3],
    linewidth=2,
    label="y2",
    color="#e5616c",
)
lines!(ax,
    dtrain.x_num[x_perm],
    pred_gaussian[x_perm, 4],
    linewidth=2,
    # label="y2",
    color="#e5616c",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("$path/multi-target-gaussian_mle-$tree_type-$device.svg", f)

