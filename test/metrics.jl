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
end
