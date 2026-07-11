using Test
using Statistics
using StatsBase: sample
using Random
using DataFrames
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

@testset "multi-target" begin

    nobs = 10_000
    Random.seed!(123)
    x_num = rand(nobs) .* 5

    y = sigmoid(logit(sin.(x_num) .* 0.5 .+ 0.5) + randn(nobs) .* 0.5)
    dtot = DataFrame(x_num=x_num, y=y)
    y = sigmoid(logit(sin.(2x_num .- 2) .* 0.5 .+ 0.5) + randn(nobs) .* 0.5)
    insertcols!(dtot, :y2 => y)

    is = sample(1:nobs, nobs, replace=false)
    ntr = floor(Int, 0.8 * nobs)
    dtrain = dtot[is[1:ntr], :]
    deval  = dtot[is[ntr+1:end], :]

    base = (var(dtrain.y), var(dtrain.y2))

    for loss in [:mse, :logloss, :gamma, :poisson, :tweedie, :mae, :quantile, :cred_std, :cred_var]
        config = EvoTreeRegressor(; loss, nrounds=200, nbins=64, L2=0.1, gamma=0.05,
            eta=0.05, max_depth=6, min_weight=1.0, rowsample=0.5, rng=123, device=:cpu)
        model = fit(config, dtrain; feature_names=["x_num"], target_name=["y", "y2"], deval, verbosity=0)
        pred = model(dtrain; device=:cpu)

        @test size(pred, 2) == 2
        @test all(isfinite, pred)
        @test mean((pred[:, 1] .- dtrain.y).^2)  < base[1] * 0.9
        @test mean((pred[:, 2] .- dtrain.y2).^2) < base[2] * 0.9
    end

    # gaussian MLE: (μ, logσ) per target → 4 columns, means at 1 and 3
    config = EvoTreeMLE(; loss=:gaussian_mle, nrounds=200, nbins=64, L2=0.1, gamma=0.05,
        eta=0.05, max_depth=6, min_weight=1.0, rowsample=0.5, rng=123, device=:cpu)
    model = fit(config, dtrain; feature_names=["x_num"], target_name=["y", "y2"], verbosity=0)
    pred = model(dtrain; device=:cpu)

    @test size(pred, 2) == 4
    @test all(isfinite, pred)
    @test mean((pred[:, 1] .- dtrain.y).^2)  < base[1] * 0.9
    @test mean((pred[:, 3] .- dtrain.y2).^2) < base[2] * 0.9
end