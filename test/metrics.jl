using Test
using Statistics
using Random
using EvoTrees
using EvoTrees: fit, predict, gini_raw, gini_norm

@testset "metrics" begin

    @testset "gini_norm" begin

        nobs = 1_000
        Random.seed!(123)
        y = rand(nobs)

        # `gini` passes a view over the prediction matrix, which is not the same concrete
        # type as `y`. This must not throw.
        p = reshape(copy(y), 1, :)
        w = ones(nobs)
        @test EvoTrees.gini(p, y, w, Float64[]) ≈ 1.0

        # A strictly monotone transform of the target ranks observations perfectly, so the
        # normalized gini is 1 regardless of the prediction's scale or offset.
        @test gini_norm(y, y) ≈ 1.0
        @test gini_norm(2 .* y .+ 10, y) ≈ 1.0
        @test gini_norm(exp.(y), y) ≈ 1.0

        # Only the ordering of the predictions matters.
        p_noisy = y .+ 0.1 .* randn(nobs)
        @test gini_norm(p_noisy, y) ≈ gini_norm(3 .* p_noisy .+ 5, y)

        # An uninformative prediction carries no ranking information.
        @test abs(gini_norm(rand(nobs), y)) < 0.1

        # A reversed ranking is the negative of a perfect one.
        @test gini_norm(-y, y) ≈ -1.0

        # Predictions are the first argument: the function is not symmetric.
        @test gini_norm(p_noisy, y) != gini_norm(y, p_noisy)

        # A better ranking scores higher than a worse one.
        @test gini_norm(p_noisy, y) > gini_norm(y .+ 1.0 .* randn(nobs), y)

        @test gini_norm([1.0], [1.0]) == 0.0
    end

    @testset "gini as eval metric" begin

        nobs = 1_000
        Random.seed!(123)
        x = rand(nobs, 5)
        y = x[:, 1] .* 2 .+ 0.1 .* randn(nobs)

        config = EvoTreeRegressor(nrounds=20, eta=0.2, metric=:gini)
        m = fit(
            config;
            x_train=x, y_train=y,
            x_eval=x, y_eval=y,
            print_every_n=1_000,
        )

        # Training must run to completion: a metric that throws, or one that never
        # improves, would error or stop early.
        @test m.info[:nrounds] == 20

        p = predict(m, x)
        @test gini_norm(p, y) > 0.8
    end

    @testset "MLE metrics" begin
        gaussian_lpdf(y, μ, σ) = -(log(σ) + (y - μ)^2 / (2 * σ^2))
        # Closed-form log-density of Logistic(μ, s).
        logistic_lpdf(y, μ, s) = -(y - μ) / s - log(s) - 2 * log1p(exp(-(y - μ) / s))

        specs = (
            (
                loss=EvoTrees.GaussianMLE,
                lpdf=gaussian_lpdf,
                fisher=function (φ, s)
                    s′ = EvoTrees.sigmoid(φ)
                    return (1 / s^2, 2 * (s′ / s)^2)
                end,
            ),
            (
                loss=EvoTrees.LogisticMLE,
                lpdf=logistic_lpdf,
                fisher=function (φ, s)
                    s′ = EvoTrees.sigmoid(φ)
                    return (1 / (3 * s^2), (π^2 + 3) / 9 * (s′ / s)^2)
                end,
            ),
        )

        @testset "$(spec.loss)" for spec in specs
            L = spec.loss

            for (μ, s) in ((0.5, 2.0), (-1.0, 0.5), (0.0, 1.0))
                φ = EvoTrees.invsoftplus(s)
                @test EvoTrees.softplus(φ) ≈ s
                for y in (μ, μ + 0.7, μ + 3.5, μ - 3.5, μ + 12.0)
                    @test EvoTrees._mle2p_metric_value(L, μ, φ, y) ≈ spec.lpdf(y, μ, s)
                end
            end

            # Metric is the log-likelihood, so its derivatives match the loss gradient up to sign.
            μ, s, h = 0.5, 2.0, 1e-6
            φ = EvoTrees.invsoftplus(s)
            for y in (1.5, 4.0, -3.0)
                g1, g2, h1, h2 = EvoTrees.mle2p_grad_hess(L, μ, φ, y)
                dμ = (EvoTrees._mle2p_metric_value(L, μ + h, φ, y) -
                      EvoTrees._mle2p_metric_value(L, μ - h, φ, y)) / (2h)
                dφ = (EvoTrees._mle2p_metric_value(L, μ, φ + h, y) -
                      EvoTrees._mle2p_metric_value(L, μ, φ - h, y)) / (2h)
                @test dμ ≈ -g1 atol = 1e-5
                @test dφ ≈ -g2 atol = 1e-5
            end

            # Fisher information does not depend on the residual, and is positive definite.
            h_ref = EvoTrees.mle2p_grad_hess(L, μ, φ, μ)[3:4]
            eh1, eh2 = spec.fisher(φ, s)
            @test h_ref[1] ≈ eh1
            @test h_ref[2] ≈ eh2
            @test h_ref[1] > 0
            @test h_ref[2] > 0
            for y in (1.5, 4.0, -3.0)
                @test EvoTrees.mle2p_grad_hess(L, μ, φ, y)[3:4] == h_ref
            end

            # Symmetric about the location, and maximised there.
            @test EvoTrees._mle2p_metric_value(L, μ, φ, μ + 2.0) ≈
                  EvoTrees._mle2p_metric_value(L, μ, φ, μ - 2.0)
            @test EvoTrees._mle2p_metric_value(L, μ, φ, μ) >
                  EvoTrees._mle2p_metric_value(L, μ, φ, μ + 2.0)
        end
    end
end
