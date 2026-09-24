using Statistics
using Random: Xoshiro
using DifferentiationInterface
using ForwardDiff

tree_structure(m) = [(t.feat, t.cond_bin, t.split) for t in m.trees]

@testset "custom loss" begin

    rng = Xoshiro(19)
    nobs = 1_000
    x = rand(rng, nobs, 4)
    y_lin = 2 .* x[:, 1] .- x[:, 2] .+ 0.3 .* randn(rng, nobs)
    y_bin = Float64.(y_lin .> mean(y_lin))
    y_pos = 0.5 .+ 3 .* rand(rng, nobs)

    shared = (nrounds=20, max_depth=4, eta=0.2, nbins=32, seed=1)
    custom(f) = EvoTreeRegressor(; loss=:custom, loss_fn=f, loss_backend=AutoForwardDiff(), shared...)

    @testset "reproduces the built-in gradient losses" begin
        cases = (
            (:mse, y_lin, (p, t) -> (p - t)^2, identity),
            (:logloss, y_bin, (p, t) -> log(1 + exp(p)) - t * p, EvoTrees.sigmoid),
            (:poisson, y_pos, (p, t) -> exp(p) - t * p, exp),
            (:gamma, y_pos, (p, t) -> 2 * (p + t * exp(-p)), exp),
            (:tweedie, y_pos, (p, t) -> 4 * (exp(p / 2) + t * exp(-p / 2)), exp),
        )
        for (loss, y, f, link) in cases
            mb = fit(EvoTreeRegressor(; loss, shared...); x_train=x, y_train=y, verbosity=0)
            mc = fit(custom(f); x_train=x, y_train=y, verbosity=0)
            @test tree_structure(mb) == tree_structure(mc)
            pb = predict(mb, x)
            pc = link.(predict(mc, x))
            @test maximum(abs, pb .- pc) < 1e-5 * max(1, maximum(abs, pb))
        end
    end

    @testset "initial prediction minimises the loss" begin
        m = fit(custom((p, t) -> (p - t)^2); x_train=x, y_train=y_lin, verbosity=0)
        @test m.bias[1] ≈ mean(y_lin) rtol = 1e-6

        m = fit(custom((p, t) -> log(1 + exp(p)) - t * p); x_train=x, y_train=y_bin, verbosity=0)
        @test m.bias[1] ≈ EvoTrees.logit(mean(y_bin)) rtol = 1e-5
    end

    @testset "a loss with no built-in equivalent" begin
        rng2 = Xoshiro(23)
        n2 = 2_000
        xo = rand(rng2, n2, 4)
        yo = 3 .* xo[:, 1] .- 2 .* xo[:, 2] .+ 0.2 .* randn(rng2, n2)
        contaminated = copy(yo)
        outlier = rand(rng2, n2) .< 0.08
        contaminated[outlier] .+= 60 .* randn(rng2, sum(outlier))

        huber = (p, t) -> (r = p - t; abs(r) <= 1 ? r^2 / 2 : abs(r) - 0.5)
        cfg = (nrounds=100, max_depth=4, eta=0.1, nbins=32, seed=1)
        mh = fit(EvoTreeRegressor(; loss=:custom, loss_fn=huber, loss_backend=AutoForwardDiff(),
                lambda=1.0, cfg...); x_train=xo, y_train=contaminated, verbosity=0)
        ms = fit(EvoTreeRegressor(; loss=:mse, cfg...); x_train=xo, y_train=contaminated, verbosity=0)
        @test mean(abs, predict(mh, xo) .- yo) < mean(abs, predict(ms, xo) .- yo)
    end

    @testset "eval metric and logger" begin
        f = (p, t) -> (p - t)^2
        m = fit(custom(f); x_train=x, y_train=y_lin, x_eval=x, y_eval=y_lin, verbosity=0)
        logger = m.info[:logger]
        @test logger[:name] == "custom"
        @test all(isfinite, logger[:metrics])
        @test logger[:metrics][end] ≈ mean(f.(predict(m, x), y_lin)) rtol = 1e-5
        @test logger[:metrics][end] < logger[:metrics][1]

        mb = fit(EvoTreeRegressor(; loss=:mse, metric=:mse, shared...);
            x_train=x, y_train=y_lin, x_eval=x, y_eval=y_lin, verbosity=0)
        @test m.info[:logger][:metrics] ≈ mb.info[:logger][:metrics] rtol = 1e-6
    end

    @testset "invalid configuration" begin
        @test_throws ErrorException EvoTreeRegressor(loss=:custom)
        @test_throws ErrorException EvoTreeRegressor(loss=:custom, loss_fn=(p, t) -> (p - t)^2, device=:gpu)
        @test_throws ErrorException fit(custom((p, t) -> (p - t)^2);
            x_train=x, y_train=hcat(y_lin, y_lin), verbosity=0)
        @test_throws ErrorException fit(
            EvoTreeRegressor(; loss=:custom, loss_fn=(p, t) -> (p - t)^2, loss_backend=nothing, shared...);
            x_train=x, y_train=y_lin, verbosity=0)
    end

end
