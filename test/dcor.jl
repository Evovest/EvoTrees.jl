using EvoTrees: dcov2, dcor2, dcov2_grad, build_ctrl, ctrl_rows, DcorCache
using Statistics: std

# Direct O(n^2) transcription of the unbiased estimator (Szekely & Rizzo 2014), used only
# to hold the O(n log n) implementation to the definition it claims to compute.
function dcov2_naive(x, y)
    n = length(x)
    a = [abs(x[i] - x[j]) for i in 1:n, j in 1:n]
    b = [abs(y[i] - y[j]) for i in 1:n, j in 1:n]
    ar = vec(sum(a, dims=2))
    br = vec(sum(b, dims=2))
    S1 = sum(a .* b)
    S2 = sum(ar .* br)
    (S1 - 2 * S2 / (n - 2) + sum(ar) * sum(br) / ((n - 1) * (n - 2))) / (n * (n - 3))
end

@testset "distance covariance" begin
    @testset "matches the definition" begin
        for (n, seed) in ((4, 9), (8, 1), (37, 2), (200, 3))
            rng = Xoshiro(seed)
            x, y = randn(rng, n), randn(rng, n)
            @test dcov2(x, y) ≈ dcov2_naive(x, y) rtol = 1e-10
            # the sweep sorts on the first argument, so symmetry holds to accumulation error
            @test dcov2(x, y) ≈ dcov2(y, x) rtol = 1e-9
        end
        # a control on an absolute scale, a timestamp or a price. Every term is a difference, so
        # an offset must not move the answer, and it does if the sums carry the offset instead of
        # being shifted first. The pair is deliberately DEPENDENT: on independent draws the
        # statistic estimates zero, and a relative tolerance against a near-zero quantity measures
        # sampling noise rather than the property under test
        rng = Xoshiro(31)
        xo = randn(rng, 200)
        yo = 0.7 .* xo .+ 0.7 .* randn(rng, 200)
        @test dcov2(xo, yo) > 0.05                     # the reference is Θ(1), not noise
        for off in (1e6, 1.7e9)
            @test dcov2(xo, yo .+ off) ≈ dcov2(xo, yo) rtol = 1e-6
            @test dcov2_grad(xo, yo .+ off) ≈ dcov2_grad(xo, yo) rtol = 1e-6
            # the prediction side carries the target's offset in production. Only the value
            # needs pinning: the gradient reads x through `sortperm` and an equality test and
            # never through its magnitude, so an offset leaves it bit-identical by construction
            # and asserting that would test nothing
            @test dcov2(xo .+ off, yo) ≈ dcov2(xo, yo) rtol = 1e-6
            @test dcov2_grad(xo .+ off, yo) == dcov2_grad(xo, yo)
        end

        # ties in both inputs, where the rank bookkeeping is easiest to get wrong
        rng = Xoshiro(4)
        xt = Float64.(rand(rng, 1:3, 60))
        yt = Float64.(rand(rng, 1:2, 60))
        @test dcov2(xt, yt) ≈ dcov2_naive(xt, yt) rtol = 1e-10
    end

    @testset "correlation properties" begin
        rng = Xoshiro(5)
        x = randn(rng, 300)
        @test dcor2(x, x) ≈ 1.0 rtol = 1e-10
        @test dcor2(x, 3 .* x .+ 7) ≈ 1.0 rtol = 1e-10
        # independent draws carry no dependence, up to estimator noise
        @test abs(dcor2(x, randn(Xoshiro(6), 300))) < 0.05
        # a symmetric relationship is invisible to Pearson and visible here. Pairing each
        # draw with its negation makes the Pearson correlation exactly zero by construction
        # rather than merely small in this sample
        u = randn(Xoshiro(8), 150)
        xs = vcat(u, -u)
        zs = abs.(xs)
        @test abs(cor(xs, zs)) < 1e-12
        @test dcor2(xs, zs) > 0.2
        # a constant carries nothing, and must not divide by zero. 2.0 is a power of two and
        # its prefix sums are exact, so it passes even when the accumulation is not shifted;
        # the other two are the cases that discriminate
        @test dcor2(x, fill(2.0, 300)) == 0.0
        @test dcor2(x, fill(0.1, 300)) == 0.0
        @test dcor2(x, fill(1 / 3, 300)) == 0.0
        @test dcov2(fill(1e6 + 0.1, 300), fill(1e6 + 0.1, 300)) == 0.0
    end

    @testset "gradient matches finite differences" begin
        rng = Xoshiro(7)
        n = 40
        x, y = randn(rng, n), randn(rng, n)
        g = dcov2_grad(x, y)
        h = 1e-6
        scale = max(maximum(abs, g), 1.0)
        for i in (1, 7, 19, n)
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            fd = (dcov2(xp, y) - dcov2(xm, y)) / (2h)
            @test abs(fd - g[i]) / scale < 1e-5
        end
        # the statistic is homogeneous of degree one in each argument, so the gradient is linear
        # in y and does not move when x is rescaled
        @test dcov2_grad(x, 2 .* y) ≈ 2 .* g rtol = 1e-10
        @test dcov2_grad(3 .* x, y) ≈ g rtol = 1e-12
    end

    @testset "tied predictions" begin
        # boosted predictions are heavily tied. At a tie the statistic has a kink, and the gradient
        # is the symmetric subgradient, which the central difference of a piecewise linear function
        # reproduces exactly for a step below the gap between tied values
        rng = Xoshiro(12)
        xt = Float64.(rand(rng, 1:4, 30))
        yc = randn(rng, 30)
        for g in (dcov2_grad(xt, yc), copy(EvoTrees.dcov2_grad!(DcorCache(yc), xt)))
            for i in 1:30
                xp = copy(xt); xp[i] += 1e-4
                xm = copy(xt); xm[i] -= 1e-4
                @test abs((dcov2(xp, yc) - dcov2(xm, yc)) / 2e-4 - g[i]) < 1e-8
            end
        end
        # so the gradient does not depend on row order, and vanishes when every prediction ties
        pm = randperm(rng, 30)
        @test dcov2_grad(xt[pm], yc[pm]) ≈ dcov2_grad(xt, yc)[pm] rtol = 1e-12
        @test all(iszero, dcov2_grad(fill(0.3, 30), yc))
    end

    @testset "input guards" begin
        @test_throws ErrorException dcov2(randn(3), randn(3))
        @test_throws ErrorException dcov2_grad(randn(3), randn(3))
        @test_throws DimensionMismatch dcov2(randn(10), randn(9))
        @test_throws DimensionMismatch dcov2_grad(randn(10), randn(9))
    end
end

@testset "decorrelation penalty" begin
    rng = Xoshiro(11)
    nobs = 2_000
    x = randn(rng, nobs, 4)
    # the control shares most of its signal with the first feature, so an unconstrained fit
    # picks up a dependence on it
    ctrl = x[:, 1] .+ 0.3 .* randn(rng, nobs)
    y = 2 .* x[:, 1] .+ x[:, 2] .+ 0.2 .* randn(rng, nobs)

    cfg(lambda) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1,
        ctrl_lambda=lambda)
    base = fit(cfg(0.0); x_train=x, y_train=y, verbosity=0)
    mid = fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
    pen = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)

    dep(m) = dcor2(Float64.(predict(m, x)[:, 1]), Float64.(ctrl))
    rmse(m) = sqrt(mean((predict(m, x)[:, 1] .- y) .^ 2))
    d_base, d_mid, d_pen = dep(base), dep(mid), dep(pen)

    @test d_pen < d_base / 3                    # the dependence is materially reduced
    @test d_base > d_mid > d_pen                # and monotonically in the penalty weight
    @test rmse(base) < rmse(mid) < rmse(pen)    # paid for in fit, as a penalty must be

    # a penalised model must still be a model: destroying the fit would satisfy the three
    # assertions above on its own, so hold it to predicting the target and to not driving the
    # statistic through zero into an anti-arrangement
    @test cor(predict(pen, x)[:, 1], y) > 0.5
    @test std(predict(pen, x)[:, 1]) > 0.25 * std(y)
    @test dcov2(Float64.(predict(pen, x)[:, 1]), Float64.(ctrl)) > 0
    # the second feature carries signal the control does not, so it must survive the penalty
    @test cor(predict(pen, x)[:, 1], x[:, 2]) > 0.2

    @testset "the default path is untouched" begin
        # no control at all, and a control with a zero weight, must both reproduce the base
        # model exactly rather than approximately
        zero_weight = fit(cfg(0.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        @test predict(zero_weight, x) == predict(base, x)
        no_ctrl = fit(EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1);
            x_train=x, y_train=y, verbosity=0)
        @test predict(no_ctrl, x) == predict(base, x)
        # every prediction ties at the start, so the first tree is exactly unpenalised
        one(λ) = EvoTreeRegressor(loss=:mse, nrounds=1, max_depth=4, eta=0.2, seed=1, ctrl_lambda=λ)
        @test predict(fit(one(10.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0), x) ==
              predict(fit(one(0.0); x_train=x, y_train=y, verbosity=0), x)
    end

    @testset "a saturating loss does not run away" begin
        # without the curvature weighting the penalty dominated the rows :logloss saturates and the
        # fit ran away within a couple of rounds, with scores blowing up and AUC below 0.5. Measured
        # on Julia 1.10 and 1.12 at this weight: dependence 0.49 unpenalised and 0.11 penalised, AUC
        # 0.94, raw score spread 5.6 unpenalised and 2.7 penalised
        yb = Float64.(y .> 0)
        cl(λ) = EvoTreeRegressor(loss=:logloss, nrounds=40, max_depth=4, eta=0.2, seed=1, ctrl_lambda=λ)
        pb = predict(fit(cl(0.0); x_train=x, y_train=yb, verbosity=0), x)[:, 1]
        pl = predict(fit(cl(10.0); x_train=x, y_train=yb, ctrl_train=ctrl, verbosity=0), x)[:, 1]
        function auc(s)
            r = invperm(sortperm(s))
            np = sum(yb); nn = length(yb) - np
            (sum(r[yb.==1]) - np * (np + 1) / 2) / (np * nn)
        end
        raw(q) = log.(q ./ (1 .- q))
        @test dcor2(Float64.(pl), Float64.(ctrl)) < dcor2(Float64.(pb), Float64.(ctrl)) / 3
        @test auc(pl) > 0.85
        @test std(raw(pl)) < std(raw(pb))
    end

    @testset "table interface" begin
        df = (f1=x[:, 1], f2=x[:, 2], f3=x[:, 3], f4=x[:, 4], c=ctrl, y=y)
        mt = fit(cfg(10.0), df; target_name="y", ctrl_name="c", verbosity=0)
        # the control is a role, not a feature, so it must not be learned from
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test mt.info[:ctrl_name] == :c
        @test dcor2(Float64.(predict(mt, df)[:, 1]), Float64.(ctrl)) < d_base / 3
    end

    @testset "the control's units do not matter" begin
        # `build_ctrl` centres and scales, so a control differing only by offset or by unit is
        # the same control. Without that, the penalty carries the column's units and the same
        # ctrl_lambda means something different for every problem: the raw gradient grows from
        # 0.006 to 168000 as the spread goes from 1 to 3e7
        @test std(build_ctrl(ctrl, nobs, "c")) ≈ 1.0
        # a change of unit is recovered exactly, so the fit is bit-identical
        @test build_ctrl(ctrl, nobs, "c") == build_ctrl(1024 .* ctrl, nobs, "c")
        @test build_ctrl(ctrl, nobs, "c") ≈ build_ctrl(3600 .* ctrl, nobs, "c") rtol = 1e-12
        # only the power of two is recovered bit for bit, so only it is compared at the model level
        mexact = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=(1024 .* ctrl), verbosity=0)
        @test predict(mexact, x) == predict(pen, x)
        # an offset is recovered only to the input's own resolution: at 1.7e9 the Float64 step
        # is 2.4e-7, so the standardised control differs by about 1e-7. It still fails loudly if
        # the control is narrowed back to Float32, where an offset collapses it to a constant and
        # `build_ctrl` throws instead
        @test build_ctrl(ctrl, nobs, "c") ≈ build_ctrl(1.7e9 .+ ctrl, nobs, "c") rtol = 1e-4
        # a one ulp change in the control can flip a split tie, so the other rescalings are held to
        # closeness at the gradient, where there is no split to flip
        p = Float32.(randn(Xoshiro(23), 1, nobs))
        ∇a = zeros(Float32, 3, nobs); ∇a[3, :] .= 1
        EvoTrees.update_grads!(∇a, p, vec(p), EvoTrees.MSE, cfg(10.0), nothing, DcorCache(build_ctrl(ctrl, nobs, "c")))
        for c2 in (3600 .* ctrl, 1.7e9 .+ ctrl)
            ∇b = zeros(Float32, 3, nobs); ∇b[3, :] .= 1
            EvoTrees.update_grads!(∇b, p, vec(p), EvoTrees.MSE, cfg(10.0), nothing, DcorCache(build_ctrl(c2, nobs, "c")))
            @test ∇b[1, :] ≈ ∇a[1, :] rtol = 1e-5
        end
        # exact distinctness is the wrong assertion: at 1.7e9 the Float64 step is 2.4e-7, so a
        # collision among 2000 draws is an ordinary birthday event and says nothing about the code
        @test length(unique(build_ctrl(1.7e9 .+ ctrl, nobs, "c"))) > 0.99 * nobs
    end

    @testset "weights do not change the penalty" begin
        # the statistic is unweighted. With y equal to p the :mse base gradient is exactly 0, so
        # row 1 holds the penalty alone and two weightings can be compared bit for bit
        wv = 0.5 .+ rand(Xoshiro(21), nobs)
        p = Float32.(randn(Xoshiro(22), 1, nobs))
        yp = vec(copy(p))
        c = build_ctrl(ctrl, nobs, "c")
        ∇u = zeros(Float32, 3, nobs); ∇u[3, :] .= 1
        ∇w = zeros(Float32, 3, nobs); ∇w[3, :] .= Float32.(wv)
        EvoTrees.update_grads!(∇u, p, yp, EvoTrees.MSE, cfg(10.0), nothing, DcorCache(c))
        EvoTrees.update_grads!(∇w, p, yp, EvoTrees.MSE, cfg(10.0), nothing, DcorCache(c))
        @test ∇w[1, :] == ∇u[1, :]
        @test Float64.(∇u[1, :]) ≈ 10.0 .* nobs .* dcov2_grad(Float64.(yp), c) rtol = 1e-5
        mw = fit(cfg(10.0); x_train=x, y_train=y, w_train=wv, ctrl_train=ctrl, verbosity=0)
        @test dcor2(Float64.(predict(mw, x)[:, 1]), Float64.(ctrl)) < d_base / 2
        # the penalty is scaled by the mean weight, so a uniform rescaling of the weights leaves the
        # fit unchanged, as it does without a control. A power of two keeps every sum exact, and
        # L2 = 0 with a matching min_weight keeps the base step itself scale free
        cw(λ, mw) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1, L2=0.0,
            min_weight=mw, ctrl_lambda=λ)
        for λ in (0.0, 10.0)
            c = λ > 0 ? ctrl : nothing
            m1 = fit(cw(λ, 8.0); x_train=x, y_train=y, w_train=ones(nobs), ctrl_train=c, verbosity=0)
            m8 = fit(cw(λ, 64.0); x_train=x, y_train=y, w_train=fill(8.0, nobs), ctrl_train=c, verbosity=0)
            @test predict(m8, x) == predict(m1, x)
        end
    end

    @testset "rejected combinations" begin
        # a weight with nothing to apply it to
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y, verbosity=0)
        # a control that cannot line up with the data
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=ctrl[1:10], verbosity=0)
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y,
            ctrl_train=fill(1.0, nobs), verbosity=0)
        # a column mixing 0.0 and -0.0 is constant, but `isequal` says otherwise, so a guard
        # written with `allequal` lets it through and the scaling then returns all NaN
        @test_throws "is constant" build_ctrl(repeat([0.0, -0.0], nobs ÷ 2), nobs, "c")
        # and a spread that overflows or underflows the variance is equally unusable
        @test_throws ErrorException build_ctrl(1e160 .* randn(Xoshiro(41), nobs), nobs, "c")
        @test_throws ErrorException build_ctrl(1e-170 .* randn(Xoshiro(42), nobs), nobs, "c")
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y,
            ctrl_train=vcat(NaN, ctrl[2:end]), verbosity=0)
        # a learner with no weight for a control must reject one rather than ignore it
        @test_throws "no `ctrl_lambda`" fit(EvoTreeClassifier(nrounds=5, max_depth=3);
            x_train=x, y_train=Int.(y .> 0), ctrl_train=ctrl, verbosity=0)
        @test_throws "no `ctrl_lambda`" fit(EvoTreeCount(nrounds=5, max_depth=3);
            x_train=x, y_train=abs.(round.(y)), ctrl_train=ctrl, verbosity=0)
        # losses whose gradient row is negated, or whose leaf reads another row, are rejected
        # rather than silently climbing the penalty
        for bad in (:mae, :quantile, :cred_var, :cred_std)
            cfgb = EvoTreeRegressor(loss=bad, nrounds=5, max_depth=3, ctrl_lambda=1.0)
            @test_throws ErrorException fit(cfgb; x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        end
        # :lambdarank subtypes the same abstract type as the supported losses, but its row 1 is a
        # pairwise lambda rather than a per-observation gradient. It needs a non-negative target
        # of its own, or the target check fires first and hides which guard is being tested
        @test_throws ErrorException fit(
            EvoTreeRegressor(loss=:lambdarank, nrounds=5, max_depth=3, ctrl_lambda=1.0);
            x_train=x, y_train=abs.(y), group_train=UInt32.((0:nobs-1) .÷ 10 .+ 1),
            ctrl_train=ctrl, verbosity=0)
        # and the ones that are supported all run, with the penalty landing on their gradient row
        # weighted by the row's curvature relative to :mse, `h / 2w`, and nowhere else
        c = build_ctrl(ctrl, nobs, "c")
        pz = Float32.(0.5 .* randn(Xoshiro(23), 1, nobs))
        for good in (:mse, :logloss, :poisson, :gamma, :tweedie)
            yy = good in (:logloss,) ? Float64.(y .> 0) : abs.(y) .+ 0.1
            cfgg = EvoTreeRegressor(loss=good, nrounds=5, max_depth=3, ctrl_lambda=2.0)
            @test all(isfinite, predict(fit(cfgg; x_train=x, y_train=yy, ctrl_train=ctrl, verbosity=0), x))
            Lg = EvoTrees._loss2type_dict[good]
            ∇a = zeros(Float32, 3, nobs); ∇a[3, :] .= 1
            ∇b = copy(∇a)
            EvoTrees.update_grads!(∇a, pz, Float32.(yy), Lg, cfgg, nothing, nothing)
            EvoTrees.update_grads!(∇b, pz, Float32.(yy), Lg, cfgg, nothing, DcorCache(c))
            expected = 2.0 .* nobs .* dcov2_grad(Float64.(vec(pz)), c) .* Float64.(∇a[2, :]) ./ 2
            @test Float64.(∇b[1, :] .- ∇a[1, :]) ≈ expected rtol = 1e-3
            @test ∇b[2:3, :] == ∇a[2:3, :]
        end
        # and the weight itself is validated like every other one
        @test_throws ErrorException EvoTreeRegressor(ctrl_lambda=-1.0)
        @test_throws ErrorException EvoTreeRegressor(ctrl_lambda=Inf)
    end
end

@testset "multi-target" begin
    @test ctrl_rows(EvoTrees.MSE, 3) == 1:3

    rng = Xoshiro(91)
    nobs = 2_000
    x = randn(rng, nobs, 4)
    ctrl = x[:, 1] .+ 0.3 .* randn(rng, nobs)

    @testset "one penalty per output" begin
        Y = hcat(2 .* x[:, 1] .+ x[:, 2], .-x[:, 1] .+ x[:, 3]) .+ 0.2 .* randn(rng, nobs, 2)
        cfg(λ) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1, ctrl_lambda=λ)
        b = fit(cfg(0.0); x_train=x, y_train=Y, verbosity=0)
        m = fit(cfg(10.0); x_train=x, y_train=Y, ctrl_train=ctrl, verbosity=0)
        for k in 1:2
            @test dcor2(Float64.(predict(m, x)[:, k]), ctrl) < dcor2(Float64.(predict(b, x)[:, k]), ctrl) / 2
        end
    end
end
