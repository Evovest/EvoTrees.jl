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
    deval = dtot[is[(ntr+1):end], :]

    base = (var(dtrain.y), var(dtrain.y2))

    for loss in [:mse, :logloss, :gamma, :poisson, :tweedie, :mae, :quantile, :cred_std, :cred_var]
        config = EvoTreeRegressor(; loss, nrounds=200, nbins=64, L2=0.1, gamma=0.05,
            eta=0.05, max_depth=6, min_weight=1.0, rowsample=0.5, seed=123, device=:cpu)
        model = fit(config, dtrain; feature_names=["x_num"], target_name=["y", "y2"], deval, verbosity=0)
        pred = model(dtrain; device=:cpu)

        @test size(pred, 2) == 2
        @test all(isfinite, pred)
        @test mean((pred[:, 1] .- dtrain.y) .^ 2) < base[1] * 0.9
        @test mean((pred[:, 2] .- dtrain.y2) .^ 2) < base[2] * 0.9
    end

    @testset "matrix y is (nobs, n_targets)" begin
        x = reshape(dtrain.x_num, :, 1)
        y = hcat(dtrain.y, dtrain.y2)
        config = EvoTreeRegressor(; loss=:mse, nrounds=50, max_depth=4, seed=123)
        model = fit(config; x_train=x, y_train=y, verbosity=0)
        pred = model(x)
        @test size(pred) == (nrow(dtrain), 2)
        @test all(isfinite, pred)
        @test mean((pred[:, 1] .- dtrain.y) .^ 2) < base[1] * 0.9
        @test mean((pred[:, 2] .- dtrain.y2) .^ 2) < base[2] * 0.9
    end

    # MLE: (μ, σ) per target → 4 columns, means at 1 and 3, positive scales at 2 and 4
    @testset for loss in [:gaussian_mle, :logistic_mle]
        config = EvoTreeMLE(; loss, nrounds=200, nbins=64, L2=0.1, gamma=0.05,
            eta=0.05, max_depth=6, min_weight=1.0, rowsample=0.5, seed=123, device=:cpu)
        model = fit(config, dtrain; feature_names=["x_num"], target_name=["y", "y2"], verbosity=0)
        pred = model(dtrain; device=:cpu)

        @test size(pred, 2) == 4
        @test all(isfinite, pred)
        @test all(>(0), pred[:, 2])
        @test all(>(0), pred[:, 4])
        @test mean((pred[:, 1] .- dtrain.y) .^ 2) < base[1] * 0.9
        @test mean((pred[:, 3] .- dtrain.y2) .^ 2) < base[2] * 0.9
    end

    # Multi-target MLE packs [mu_1, phi_1, mu_2, phi_2, ...], so every even offset column
    # is a scale and must be unconstrained, not just column 2. The eval callback builds
    # its own copy of this mapping, so it is worth pinning that it covers all of them.
    @testset "multi-target MLE eval offset" begin
        n_off = 200
        rng_off = Xoshiro(9)
        x_off = reshape(rand(rng_off, n_off), :, 1)
        Y_off = hcat(rand(rng_off, n_off), rand(rng_off, n_off))
        s1, s2 = 1.2, 1.5
        off = hcat(fill(0.1, n_off), fill(s1, n_off), fill(-0.2, n_off), fill(s2, n_off))

        cfg = EvoTreeMLE(; loss=:gaussian_mle, nrounds=0, max_depth=3, metric=:gaussian_mle)
        m_off = fit(cfg; x_train=x_off, y_train=Y_off, offset_train=copy(off), verbosity=0)
        cb = EvoTrees.CallBack(cfg, m_off, x_off, Y_off, EvoTrees.CPU; offset_eval=copy(off))

        @test cb.p[2, 1] ≈ log(s1) atol = 1e-5
        @test cb.p[4, 1] ≈ log(s2) atol = 1e-5

        # A single-target offset has only two columns, and must be unchanged by `2:2:end`.
        off1 = hcat(fill(0.1, n_off), fill(s1, n_off))
        m1 = fit(cfg; x_train=x_off, y_train=Y_off[:, 1], offset_train=copy(off1), verbosity=0)
        cb1 = EvoTrees.CallBack(cfg, m1, x_off, Y_off[:, 1], EvoTrees.CPU; offset_eval=copy(off1))
        @test cb1.p[2, 1] ≈ log(s1) atol = 1e-5
    end
end
