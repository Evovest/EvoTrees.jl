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

    @testset "logistic_mle metric" begin

        # Closed-form log-density of the logistic distribution with location mu and
        # scale s: -(y-mu)/s - log(s) - 2*log(1 + exp(-(y-mu)/s)).
        true_lpdf(y, mu, s) = -(y - mu) / s - log(s) - 2 * log1p(exp(-(y - mu) / s))

        for (mu, s) in ((0.5, 2.0), (-1.0, 0.5), (0.0, 1.0))
            ls = log(s)
            for y in (mu, mu + 0.7, mu + 3.5, mu - 3.5, mu + 12.0)
                @test EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu, ls, y) ≈
                      true_lpdf(y, mu, s)
            end
        end

        # The metric is the log-likelihood of the objective being optimised, so its
        # derivative must agree with the gradient the loss uses, up to sign.
        mu, ls, h = 0.5, log(2.0), 1e-6
        for y in (1.5, 4.0, -3.0)
            d = (EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu + h, ls, y) -
                 EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu - h, ls, y)) / (2h)
            g = EvoTrees.mle2p_grad_hess(EvoTrees.LogisticMLE, mu, ls, y)[1]
            @test d ≈ -g atol = 1e-5
        end

        # Symmetric about the location, and maximised there.
        mu, ls = 0.5, log(2.0)
        @test EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu, ls, mu + 2.0) ≈
              EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu, ls, mu - 2.0)
        @test EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu, ls, mu) >
              EvoTrees._mle2p_metric_value(EvoTrees.LogisticMLE, mu, ls, mu + 2.0)
    end

    @testset "mlogloss gradient and hessian" begin

        # Softmax cross-entropy for a single observation with target class k.
        loss(p, k) = log(sum(exp, p)) - p[k]

        # Equal logits are the initialisation state, and are not on their own a
        # sufficient check: the diagonal hessian prob*(1-prob) and the expression
        # (1-prob)/isum coincide exactly when every logit is equal. The points below
        # are deliberately unequal so that the two disagree.
        target = 2
        for p in ([0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, -0.5, 0.3], [-1.0, -1.0, 2.0])
            isum = sum(exp, p)
            h = 1e-5
            for k in eachindex(p)
                g, hess = EvoTrees.mlogloss_grad_hess(p[k], isum, k == target)
                pp = copy(p)
                pp[k] += h
                pm = copy(p)
                pm[k] -= h
                @test g ≈ (loss(pp, target) - loss(pm, target)) / (2h) atol = 1e-5
                @test hess ≈
                      (loss(pp, target) - 2 * loss(p, target) + loss(pm, target)) / h^2 atol =
                    1e-4
            end
        end
    end
end
