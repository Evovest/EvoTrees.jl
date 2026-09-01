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

    @testset "quantile metric alpha" begin

        # The eval callback must pass the model's own `alpha` to the metric. It used to
        # leave `metric_kwargs` empty for `:quantile`, so the metric fell back to its
        # `alpha = 0.5` default and always reported the median pinball loss.
        pinball(p, y, a) = mean(@. a * max(y - p, 0) + (1 - a) * max(p - y, 0))

        Random.seed!(3)
        nobs = 4_000
        x = rand(nobs, 3)
        y = 2 .* x[:, 1] .+ 0.5 .* randn(nobs)
        itr, ite = 1:3_000, 3_001:nobs

        # alpha = 0.5 is not on its own a sufficient check: it is the metric's default,
        # so it agrees whether or not alpha is plumbed through.
        for alpha in (0.1, 0.5, 0.9)
            m = fit(
                EvoTreeRegressor(;
                    loss=:quantile, alpha, nrounds=25, max_depth=4, metric=:quantile);
                x_train=x[itr, :], y_train=y[itr],
                x_eval=x[ite, :], y_eval=y[ite], verbosity=0)
            reported = m.info[:logger][:metrics][end]
            p = predict(m, x[ite, :])
            @test reported ≈ pinball(p, y[ite], alpha) rtol = 1e-4
        end
    end

    @testset "eval class encoding" begin

        # The eval callback must encode `y_eval` against the levels the model was trained
        # on. It used to derive them from `y_eval` itself, so an eval fold missing a
        # training class was scored against the wrong prediction columns.
        Random.seed!(1)
        nobs = 4_000
        x = rand(nobs, 3)
        z = 2 .* x[:, 1] .- x[:, 2]
        y = [zi < -0.3 ? 1 : (zi < 0.5 ? 2 : 3) for zi in z]

        # mlogloss computed directly, mapping each label through the model's own levels.
        function true_mlogloss(m, xe, ye)
            p = predict(m, xe)
            lv = m.info[:target_levels]
            mean(-log(max(p[i, findfirst(==(ye[i]), lv)], 1e-15)) for i in eachindex(ye))
        end

        # A class-complete eval set agrees either way; the missing-class cases are what
        # separate the two encodings.
        for ie in (1:1_500, findall(y[1:2_000] .!= 1), findall(y[1:2_000] .== 3))
            m = fit(
                EvoTreeClassifier(; nrounds=20, max_depth=4, eta=0.1);
                x_train=x, y_train=y,
                x_eval=x[ie, :], y_eval=y[ie], verbosity=0)
            @test m.info[:logger][:metrics][end] ≈ true_mlogloss(m, x[ie, :], y[ie]) rtol =
                1e-4
        end

        # A level in `y_eval` that never appeared in `y_train` indexed past the end of the
        # prediction matrix under `@inbounds`. It must be rejected instead.
        tr = findall(y .!= 3)
        @test_throws ErrorException fit(
            EvoTreeClassifier(; nrounds=5, max_depth=3);
            x_train=x[tr, :], y_train=y[tr],
            x_eval=x[1:500, :], y_eval=y[1:500], verbosity=0)
    end

    @testset "MLE metrics" begin
        gaussian_lpdf(y, loc, scale) = -(log(scale) + (y - loc)^2 / (2 * scale^2))
        logistic_lpdf(y, loc, scale) = -(y - loc) / scale - log(scale) - 2 * log1p(exp(-(y - loc) / scale))

        specs = (
            (
                loss=EvoTrees.GaussianMLE,
                lpdf=gaussian_lpdf,
                # Observed Hessian (≤0.18.7): h_loc = 1/σ², h_scale = 2 resid²/σ².
                hess=function (scale, resid)
                    return (1 / scale^2, 2 * resid^2 / scale^2)
                end,
                fisher=false,
            ),
            (
                loss=EvoTrees.LogisticMLE,
                lpdf=logistic_lpdf,
                # Fisher information: independent of residual, positive definite.
                hess=function (scale, _resid)
                    return (1 / (3 * scale^2), (π^2 + 3) / 9)
                end,
                fisher=true,
            ),
        )

        @testset "$(spec.loss)" for spec in specs
            L = spec.loss

            for (loc, scale) in ((0.5, 2.0), (-1.0, 0.5), (0.0, 1.0))
                scale_raw = log(scale)
                @test exp(scale_raw) ≈ scale
                for y in (loc, loc + 0.7, loc + 3.5, loc - 3.5, loc + 12.0)
                    @test EvoTrees._mle2p_metric_value(L, loc, scale_raw, y) ≈ spec.lpdf(y, loc, scale)
                end
            end

            # Metric is the log-likelihood, so its derivatives match the loss gradient up to sign.
            loc, scale, h = 0.5, 2.0, 1e-6
            scale_raw = log(scale)
            for y in (1.5, 4.0, -3.0)
                g1, g2, h1, h2 = EvoTrees.mle2p_grad_hess(L, loc, scale_raw, y)
                dloc = (EvoTrees._mle2p_metric_value(L, loc + h, scale_raw, y) -
                        EvoTrees._mle2p_metric_value(L, loc - h, scale_raw, y)) / (2h)
                draw = (EvoTrees._mle2p_metric_value(L, loc, scale_raw + h, y) -
                        EvoTrees._mle2p_metric_value(L, loc, scale_raw - h, y)) / (2h)
                @test dloc ≈ -g1 atol = 1e-5
                @test draw ≈ -g2 atol = 1e-5

                eh1, eh2 = spec.hess(scale, loc - y)
                @test h1 ≈ eh1
                @test h2 ≈ eh2
            end

            if spec.fisher
                # Fisher information does not depend on the residual, and is positive definite.
                h_ref = EvoTrees.mle2p_grad_hess(L, loc, scale_raw, loc)[3:4]
                eh1, eh2 = spec.hess(scale, 0.0)
                @test h_ref[1] ≈ eh1
                @test h_ref[2] ≈ eh2
                @test h_ref[1] > 0
                @test h_ref[2] > 0
                for y in (1.5, 4.0, -3.0)
                    @test EvoTrees.mle2p_grad_hess(L, loc, scale_raw, y)[3:4] == h_ref
                end
            else
                # Observed Hessian for Gaussian scale depends on the residual.
                h_at_loc = EvoTrees.mle2p_grad_hess(L, loc, scale_raw, loc)[3:4]
                @test h_at_loc[1] ≈ 1 / scale^2
                @test h_at_loc[2] ≈ 0
                h_away = EvoTrees.mle2p_grad_hess(L, loc, scale_raw, loc + 3.0)[3:4]
                @test h_away[1] ≈ h_at_loc[1]
                @test h_away[2] > h_at_loc[2]
            end

            # Symmetric about the location, and maximised there.
            @test EvoTrees._mle2p_metric_value(L, loc, scale_raw, loc + 2.0) ≈
                  EvoTrees._mle2p_metric_value(L, loc, scale_raw, loc - 2.0)
            @test EvoTrees._mle2p_metric_value(L, loc, scale_raw, loc) >
                  EvoTrees._mle2p_metric_value(L, loc, scale_raw, loc + 2.0)
        end

        # Shared CPU/GPU inverse-link: unconstrained scale → exp.
        pred = Float32[0.0 1.0; -2.0 0.5]
        EvoTrees.apply_prediction_link!(pred, EvoTrees.GaussianMLE)
        @test pred[1, :] == Float32[0.0, 1.0]
        @test pred[2, :] ≈ exp.(Float32[-2.0, 0.5])
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

    @testset "poisson metric with zero counts" begin

        for pk in (-1.0, 0.0, 0.5)
            @test EvoTrees._metric_value(EvoTrees.Poisson, pk, 0.0, 0.5) ≈ 2 * exp(pk)
        end

        dev(y, mu) = 2 * (y * log(y / mu) + mu - y)
        for (pk, y) in ((0.0, 1.0), (0.5, 3.0), (-0.7, 2.0))
            @test EvoTrees._metric_value(EvoTrees.Poisson, pk, y, 0.5) ≈ dev(y, exp(pk))
        end
        @test EvoTrees._metric_value(EvoTrees.Poisson, log(2.0), 2.0, 0.5) ≈ 0 atol = 1e-12

        p = zeros(1, 3)
        @test isfinite(EvoTrees.poisson(p, [0.0, 1.0, 2.0], ones(3), zeros(3)))

        nobs = 1_000
        Random.seed!(123)
        x = rand(nobs, 5)
        y = Float64.(rand(0:3, nobs))
        @test any(iszero, y)

        config = EvoTreeCount(nrounds=20, eta=0.2, early_stopping_rounds=3)
        m = fit(config; x_train=x, y_train=y, x_eval=x, y_eval=y, verbosity=0)
        @test all(isfinite, m.info[:logger][:metrics])
        @test m.info[:nrounds] == 20
    end
end
