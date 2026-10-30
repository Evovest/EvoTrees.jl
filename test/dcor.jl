using EvoTrees: dcov2, dcor2, dcov2_grad, build_ctrl, ctrl_rows, DcorCache, GroupedDcorCache,
    build_group_index, _penalize_row!
using Statistics: std

# O(n^2) unbiased estimator (Szekely & Rizzo 2014), the reference for the O(n log n) one.
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
            # symmetric up to accumulation error
            @test dcov2(x, y) ≈ dcov2(y, x) rtol = 1e-9
        end
        # an offset, as on a timestamp, leaves the value unchanged. The pair is dependent, so the
        # reference is not near zero
        rng = Xoshiro(31)
        xo = randn(rng, 200)
        yo = 0.7 .* xo .+ 0.7 .* randn(rng, 200)
        @test dcov2(xo, yo) > 0.05
        for off in (1e6, 1.7e9)
            @test dcov2(xo, yo .+ off) ≈ dcov2(xo, yo) rtol = 1e-6
            @test dcov2_grad(xo, yo .+ off) ≈ dcov2_grad(xo, yo) rtol = 1e-6
            # on the prediction side the gradient depends only on the order, so it is unchanged exactly
            @test dcov2(xo .+ off, yo) ≈ dcov2(xo, yo) rtol = 1e-6
            @test dcov2_grad(xo .+ off, yo) == dcov2_grad(xo, yo)
        end

        # ties in both inputs
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
        # independent draws, up to estimator noise
        @test abs(dcor2(x, randn(Xoshiro(6), 300))) < 0.05
        # a symmetric relationship with a Pearson correlation of exactly zero
        u = randn(Xoshiro(8), 150)
        xs = vcat(u, -u)
        zs = abs.(xs)
        @test abs(cor(xs, zs)) < 1e-12
        @test dcor2(xs, zs) > 0.2
        # a constant gives 0, including values whose prefix sums are inexact
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
        # linear in y, and unchanged when x is rescaled
        @test dcov2_grad(x, 2 .* y) ≈ 2 .* g rtol = 1e-10
        @test dcov2_grad(3 .* x, y) ≈ g rtol = 1e-12
    end

    @testset "tied predictions" begin
        # at ties the gradient is the symmetric subgradient, which a central difference with a step
        # below the gap between values reproduces
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
        # independent of row order, and zero when every prediction ties
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
    # the control tracks the first feature
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

    @test d_pen < d_base / 3
    @test d_base > d_mid > d_pen
    @test rmse(base) < rmse(mid) < rmse(pen)

    # the penalised model still fits the target and does not overshoot to a negative dcov2
    @test cor(predict(pen, x)[:, 1], y) > 0.5
    @test std(predict(pen, x)[:, 1]) > 0.25 * std(y)
    @test dcov2(Float64.(predict(pen, x)[:, 1]), Float64.(ctrl)) > 0
    # the signal independent of the control remains
    @test cor(predict(pen, x)[:, 1], x[:, 2]) > 0.2

    @testset "the default path is untouched" begin
        # no control, or a zero weight, gives the base model exactly
        zero_weight = fit(cfg(0.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        @test predict(zero_weight, x) == predict(base, x)
        no_ctrl = fit(EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1);
            x_train=x, y_train=y, verbosity=0)
        @test predict(no_ctrl, x) == predict(base, x)
        # predictions all tie at the start, so the first tree is unpenalised
        one_round(λ) = EvoTreeRegressor(loss=:mse, nrounds=1, max_depth=4, eta=0.2, seed=1, ctrl_lambda=λ)
        @test predict(fit(one_round(10.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0), x) ==
              predict(fit(one_round(0.0); x_train=x, y_train=y, verbosity=0), x)
    end

    @testset "a saturating loss does not run away" begin
        # the curvature weighting keeps :logloss stable
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
        # the control is not a feature
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test mt.info[:ctrl_names] == [:c]
        @test dcor2(Float64.(predict(mt, df)[:, 1]), Float64.(ctrl)) < d_base / 3
    end

    @testset "the control's units do not matter" begin
        # controls are standardised, so their offset and units do not matter
        @test std(build_ctrl(ctrl, nobs, "c")) ≈ 1.0
        # a change of unit is exact for a power of two, so the fit is identical
        @test build_ctrl(ctrl, nobs, "c") == build_ctrl(1024 .* ctrl, nobs, "c")
        @test build_ctrl(ctrl, nobs, "c") ≈ build_ctrl(3600 .* ctrl, nobs, "c") rtol = 1e-12
        mexact = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=(1024 .* ctrl), verbosity=0)
        @test predict(mexact, x) == predict(pen, x)
        # an offset is recovered to the input's resolution
        @test build_ctrl(ctrl, nobs, "c") ≈ build_ctrl(1.7e9 .+ ctrl, nobs, "c") rtol = 1e-4
        # near 1e12 the spread is still the column's own
        q = ctrl .+ 1e12
        @test diff(build_ctrl(q, nobs, "c")) ≈ diff(build_ctrl(q .- 1e12, nobs, "c")) rtol = 1e-9
        # other rescalings can flip a split tie, so they are compared at the gradient
        p = Float32.(randn(Xoshiro(23), 1, nobs))
        ∇a = zeros(Float32, 3, nobs); ∇a[3, :] .= 1
        EvoTrees.update_grads!(∇a, p, vec(p), EvoTrees.MSE, cfg(10.0), nothing, DcorCache(build_ctrl(ctrl, nobs, "c")))
        for c2 in (3600 .* ctrl, 1.7e9 .+ ctrl)
            ∇b = zeros(Float32, 3, nobs); ∇b[3, :] .= 1
            EvoTrees.update_grads!(∇b, p, vec(p), EvoTrees.MSE, cfg(10.0), nothing, DcorCache(build_ctrl(c2, nobs, "c")))
            @test ∇b[1, :] ≈ ∇a[1, :] rtol = 1e-5
        end
        # at 1.7e9 a few collisions among 2000 draws are expected
        @test length(unique(build_ctrl(1.7e9 .+ ctrl, nobs, "c"))) > 0.99 * nobs
    end

    @testset "weights do not change the penalty" begin
        # the statistic is unweighted. With y = p, row 1 holds the penalty alone
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
        # a uniform rescaling of the weights leaves the fit unchanged. L2 = 0 and a matching
        # min_weight keep the base step scale free
        cw(λ, mw) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1, L2=0.0,
            min_weight=mw, ctrl_lambda=λ)
        for λ in (0.0, 10.0)
            c = λ > 0 ? ctrl : nothing
            m1 = fit(cw(λ, 8.0); x_train=x, y_train=y, w_train=ones(nobs), ctrl_train=c, verbosity=0)
            m8 = fit(cw(λ, 64.0); x_train=x, y_train=y, w_train=fill(8.0, nobs), ctrl_train=c, verbosity=0)
            @test predict(m8, x) == predict(m1, x)
        end
        # the same within groups, each group weighing by its total weight
        gid = repeat(1:40, inner=nobs ÷ 40)
        cg(mw) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1, L2=0.0,
            min_weight=mw, ctrl_lambda=10.0, ctrl_within_group=true)
        g1 = fit(cg(8.0); x_train=x, y_train=y, w_train=ones(nobs), ctrl_train=ctrl, group_train=gid, verbosity=0)
        g8 = fit(cg(64.0); x_train=x, y_train=y, w_train=fill(8.0, nobs), ctrl_train=ctrl, group_train=gid,
            verbosity=0)
        @test predict(g8, x) == predict(g1, x)
        gc = EvoTrees.dcor_cache(cg(8.0), build_ctrl(ctrl, nobs, "c"), build_group_index(gid, nobs, "g"), wv)
        @test gc.caches[1].Wg ≈ [sum(wv[gid.==g]) for g in 1:40] rtol = 1e-14
    end

    @testset "rejected combinations" begin
        # a weight without a control
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y, verbosity=0)
        # a control of the wrong length, or constant
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=ctrl[1:10], verbosity=0)
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y,
            ctrl_train=fill(1.0, nobs), verbosity=0)
        # 0.0 and -0.0 count as equal
        @test_throws "is constant" build_ctrl(repeat([0.0, -0.0], nobs ÷ 2), nobs, "c")
        # a spread that overflows or underflows the variance, and a NaN
        @test_throws ErrorException build_ctrl(1e160 .* randn(Xoshiro(41), nobs), nobs, "c")
        @test_throws ErrorException build_ctrl(1e-170 .* randn(Xoshiro(42), nobs), nobs, "c")
        @test_throws ErrorException fit(cfg(1.0); x_train=x, y_train=y,
            ctrl_train=vcat(NaN, ctrl[2:end]), verbosity=0)
        # learners without `ctrl_lambda`
        @test_throws "no `ctrl_lambda`" fit(EvoTreeClassifier(nrounds=5, max_depth=3);
            x_train=x, y_train=Int.(y .> 0), ctrl_train=ctrl, verbosity=0)
        @test_throws "no `ctrl_lambda`" fit(EvoTreeCount(nrounds=5, max_depth=3);
            x_train=x, y_train=abs.(round.(y)), ctrl_train=ctrl, verbosity=0)
        # losses whose gradient row is negated, or whose leaf reads another row
        for bad in (:mae, :quantile, :cred_var, :cred_std)
            cfgb = EvoTreeRegressor(loss=bad, nrounds=5, max_depth=3, ctrl_lambda=1.0)
            @test_throws ErrorException fit(cfgb; x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        end
        # :lambdarank, whose row 1 is a pairwise lambda, with a valid target so this guard is reached
        @test_throws ErrorException fit(
            EvoTreeRegressor(loss=:lambdarank, nrounds=5, max_depth=3, ctrl_lambda=1.0);
            x_train=x, y_train=abs.(y), group_train=UInt32.((0:nobs-1) .÷ 10 .+ 1),
            ctrl_train=ctrl, verbosity=0)
        # the supported losses add the penalty to the gradient row only, weighted by `h / 2w`
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
        # invalid weights
        @test_throws ErrorException EvoTreeRegressor(ctrl_lambda=-1.0)
        @test_throws ErrorException EvoTreeRegressor(ctrl_lambda=Inf)
        @test_throws ErrorException EvoTreeMLE(ctrl_lambda=Inf)
        # a weight reassigned after construction is checked at fit
        for bad in (-1.0, NaN, Inf)
            mr = cfg(1.0)
            mr.ctrl_lambda = bad
            @test_throws "ctrl_lambda" fit(mr; x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        end
        @test_throws "ctrl_within_group" EvoTreeRegressor(ctrl_within_group=1)
        @test_throws "missing values" build_ctrl(Union{Missing,Float64}[1.0, missing, 2.0, 3.0, 4.0], 5, "c")
        # no distance variance: an indicator on one row, or one row either side of a tie
        ind = zeros(nobs)
        ind[7] = 1.0
        @test_throws "tied but at most one" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=ind, verbosity=0)
        ind[9] = -2.0
        @test_throws "tied but at most one" build_ctrl(ind, nobs, "c")
        # two rows on the same side are a real spread
        ind[9] = 2.0
        @test build_ctrl(ind, nobs, "c") isa Vector{Float64}
        # ctrl_within_group without groups
        @test_throws "no groups were given" fit(
            EvoTreeRegressor(loss=:mse, nrounds=5, max_depth=3, ctrl_lambda=1.0, ctrl_within_group=true);
            x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
    end
end

@testset "within-group penalty" begin
    @testset "gradient matches the per-group definition" begin
        # six interleaved groups: one of exactly 4 rows is kept; one of 3 rows and one with a constant
        # control are skipped
        rng = Xoshiro(71)
        sizes = (9, 12, 4, 3, 10, 8)
        gid = vcat([fill(g, m) for (g, m) in enumerate(sizes)]...)
        perm = randperm(rng, length(gid))
        gid = gid[perm]
        n = length(gid)
        ctrl = randn(rng, n)
        ctrl[gid.==5] .= 0.7
        p = randn(rng, n)
        λ = 2.0
        cache = GroupedDcorCache(ctrl, build_group_index(gid, n, "g"))
        @test length(cache.caches) == 4

        # an :mse curvature weighs the penalty by 1, added to the base gradient
        hrow, wrow = fill(2.0, n), ones(n)
        base_g = randn(Xoshiro(72), n)
        grow = copy(base_g)
        _penalize_row!(grow, hrow, wrow, p, cache, λ)
        grow .-= base_g
        expected = zeros(n)
        for g in (1, 2, 3, 6)
            r = findall(==(g), gid)
            z = (ctrl[r] .- mean(ctrl[r])) ./ std(ctrl[r])
            expected[r] .= λ .* length(r) .* dcov2_grad(p[r], z)
        end
        @test grow ≈ expected rtol = 1e-10
        @test all(iszero, grow[gid.==4]) && all(iszero, grow[gid.==5])
        # another curvature scales each row by `h / 2w`
        hrow2 = 0.5 .+ rand(Xoshiro(73), n)
        grow3 = zeros(n)
        _penalize_row!(grow3, hrow2, wrow, p, cache, λ)
        @test grow3 ≈ expected .* hrow2 ./ 2 rtol = 1e-10

        # finite differences of the objective
        function F(pp)
            tot = 0.0
            for g in (1, 2, 3, 6)
                r = findall(==(g), gid)
                z = (ctrl[r] .- mean(ctrl[r])) ./ std(ctrl[r])
                tot += λ * length(r) * dcov2(pp[r], z)
            end
            tot
        end
        h = 1e-6
        for i in (1, 11, 23, n)
            pp = copy(p); pp[i] += h
            pm = copy(p); pm[i] -= h
            @test abs((F(pp) - F(pm)) / (2h) - grow[i]) < 1e-5 * max(1.0, maximum(abs, grow))
        end

        # a second call gives the same result
        grow2 = copy(base_g)
        _penalize_row!(grow2, hrow, wrow, p, cache, λ)
        @test grow2 .- base_g == grow

        # no usable group is an error
        @test_throws "No group" GroupedDcorCache(ctrl[1:6], build_group_index([1, 1, 1, 2, 2, 2], 6, "g"))
    end

    @testset "groups without a usable spread are skipped" begin
        # six groups of 8 rows: a real spread, an indicator on one row, one row off either side of a
        # tie, two values one ulp apart, a spread that would underflow the scaling, a real spread
        rng = Xoshiro(74)
        n = 48
        gid = repeat(1:6, inner=8)
        c = randn(rng, n)
        c[9:16] .= 0.0
        c[12] = 1.0
        c[17:24] .= 0.3
        c[18] = -1.0
        c[23] = 2.0
        c[25:32] .= repeat([1.0, nextfloat(1.0)], 4)
        c[33:40] .= repeat([0.0, 1e-300], 4)
        cache = GroupedDcorCache(c, build_group_index(gid, n, "g"))
        @test cache.rows == [collect(1:8), collect(41:48)]
        @test all(dc -> all(isfinite, dc.ctrl), cache.caches)
    end

    @testset "each group weighs by its total weight" begin
        # weights vary within and between groups; group g adds λ W_g dcov2(p_g, c_g). With y = p,
        # row 1 holds the penalty alone
        rng = Xoshiro(76)
        sizes = (30, 45, 60, 25, 50, 40)
        gid = vcat([fill(g, m) for (g, m) in enumerate(sizes)]...)
        gid = gid[randperm(rng, length(gid))]
        n = length(gid)
        c = randn(rng, n)
        p = 0.4 .* c .+ randn(rng, n)
        w = exp.(randn(rng, length(sizes)))[gid] .* (0.5 .+ rand(rng, n))
        gi = build_group_index(gid, n, "g")
        cb = build_ctrl(c, n, "c")
        λ = 3.0
        params = EvoTreeRegressor(loss=:mse, ctrl_lambda=λ, ctrl_within_group=true)
        function penalty(wv)
            ∇ = zeros(3, n); ∇[3, :] .= wv
            EvoTrees.update_grads!(∇, reshape(p, 1, n), p, EvoTrees.MSE, params, nothing,
                EvoTrees.dcor_cache(params, cb, gi, wv))
            return ∇[1, :]
        end
        groups = [findall(==(g), gid) for g in eachindex(sizes)]
        z(r) = (c[r] .- mean(c[r])) ./ std(c[r])
        ref = zeros(n)
        for r in groups
            ref[r] .= λ * sum(w[r]) .* dcov2_grad(p[r], z(r))
        end
        pw = penalty(w)
        @test pw ≈ ref rtol = 1e-10
        # scaling one group's weights scales its penalty alone, exactly for a power of two
        w4 = copy(w)
        w4[groups[2]] .*= 4
        p4 = penalty(w4)
        @test p4[groups[2]] == 4 .* pw[groups[2]]
        @test p4[gid.!=2] == pw[gid.!=2]
        # the eval term weighs each group by its share of the eval weight
        ec = EvoTrees.eval_ctrl(params, EvoTrees.MSE, 1, EvoTrees.Controls([cb], [1.0], ["c"]), gi,
            EvoTrees.mse, w)
        dep, term = EvoTrees.eval_penalty!(ec, reshape(p, 1, n))
        d = sum(sum(w[r]) / sum(w) * dcov2(p[r], z(r)) for r in groups)
        @test dep[1] ≈ d rtol = 1e-8
        @test term ≈ λ * d rtol = 1e-8
        # unit weights give W_g the number of rows
        gc = EvoTrees.dcor_cache(params, cb, gi, ones(Float32, n)).caches[1]
        @test gc.Wg == length.(gc.rows)
        before = zeros(n)
        for (k, r) in enumerate(gc.rows)
            before[r] .= λ * length(r) .* EvoTrees.dcov2_grad!(gc.caches[k], view(p, r))
        end
        @test penalty(ones(n)) == before
    end

    @testset "the within-group penalty keeps the group-level relationship" begin
        # 100 shuffled groups of 50 rows. The control and the target share a group-level term, which
        # the within-group penalty keeps and the pooled one removes
        rng = Xoshiro(81)
        ngroups, group_size = 100, 50
        nobs = ngroups * group_size
        group = repeat(1:ngroups, inner=group_size)
        level = randn(rng, ngroups)[group]
        dev = randn(rng, nobs)
        x = hcat(level .+ 0.1 .* randn(rng, nobs), dev .+ 0.3 .* randn(rng, nobs), randn(rng, nobs, 2))
        ctrl = level .+ dev
        y = level .+ dev .+ x[:, 3] .+ 0.3 .* randn(rng, nobs)
        perm = randperm(Xoshiro(82), nobs)
        group, x, ctrl, y = group[perm], x[perm, :], ctrl[perm], y[perm]

        cfgp(λ, within) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1,
            ctrl_lambda=λ, ctrl_within_group=within)
        base = fit(cfgp(0.0, false); x_train=x, y_train=y, verbosity=0)
        pooled = fit(cfgp(20.0, false); x_train=x, y_train=y, ctrl_train=ctrl, group_train=group, verbosity=0)
        within = fit(cfgp(20.0, true); x_train=x, y_train=y, ctrl_train=ctrl, group_train=group, verbosity=0)

        rows = [findall(==(g), group) for g in 1:ngroups]
        pred(m) = Float64.(predict(m, x)[:, 1])
        dep_within(p) = mean(dcor2(p[r], ctrl[r]) for r in rows)
        groupmean(v) = [mean(v[r]) for r in rows]
        dep_between(p) = dcor2(groupmean(p), groupmean(ctrl))
        pb, pp, pw = pred(base), pred(pooled), pred(within)

        @test dep_within(pw) < dep_within(pb) / 3
        # the group-level relationship survives the within-group penalty and not the pooled one
        @test dep_between(pw) > 0.85
        @test dep_between(pp) < 0.75
        @test cor(pw, y) > 0.5

        # a zero weight leaves the model untouched
        base_g = fit(cfgp(0.0, false); x_train=x, y_train=y, group_train=group, verbosity=0)
        @test predict(fit(cfgp(0.0, true); x_train=x, y_train=y, ctrl_train=ctrl, group_train=group,
            verbosity=0), x) == predict(base_g, x)

        # whole-group sampling and the table interface
        @test fit(EvoTreeRegressor(loss=:mse, nrounds=10, max_depth=3, rowsample=0.5, ctrl_lambda=5.0,
            ctrl_within_group=true); x_train=x, y_train=y, ctrl_train=ctrl, group_train=group,
            verbosity=0) isa EvoTrees.EvoTree
        df = (f1=x[:, 1], f2=x[:, 2], f3=x[:, 3], f4=x[:, 4], g=group, c=ctrl, y=y)
        mt = fit(cfgp(20.0, true), df; target_name="y", group_name="g", ctrl_name="c", verbosity=0)
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test dep_within(Float64.(predict(mt, df)[:, 1])) < dep_within(pb) / 3
    end
end

@testset "multi-target and likelihood losses" begin
    @test ctrl_rows(EvoTrees.MSE, 3) == 1:3
    @test collect(ctrl_rows(EvoTrees.GaussianMLE, 4)) == [1, 3]

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

        # grouped and multi-target: each output row gets its own within-group term. With y = p,
        # row k holds the penalty alone
        gid = repeat(1:40, inner=50)
        gc = GroupedDcorCache(build_ctrl(ctrl, nobs, "c"), build_group_index(gid, nobs, "g"))
        P2 = Float32.(randn(Xoshiro(92), 2, nobs))
        ∇g = zeros(Float32, 5, nobs); ∇g[5, :] .= 1
        EvoTrees.update_grads!(∇g, P2, copy(P2), EvoTrees.MSE, cfg(3.0), nothing, gc)
        for k in 1:2
            ref = zeros(nobs)
            _penalize_row!(ref, fill(2.0, nobs), ones(nobs), Float64.(P2[k, :]), gc, 3.0)
            @test Float64.(∇g[k, :]) ≈ ref rtol = 1e-5
        end
        @test all(==(2), ∇g[3:4, :])
    end

    @testset "the location moves and the scale does not" begin
        y = 2 .* x[:, 1] .+ x[:, 2] .+ 0.2 .* randn(rng, nobs)
        # only the location rows get the penalty, weighted by `h / 2w = exp(-2 * scale_raw) / 2`. With
        # y equal to the location, row 1 holds the penalty alone
        params = EvoTreeMLE(loss=:gaussian_mle, ctrl_lambda=1.0)
        c = build_ctrl(ctrl, nobs, "c")
        curv(sraw) = exp.(-2 .* Float64.(sraw)) ./ 2
        p = Float32.(vcat(randn(rng, 1, nobs), 0.1f0 .* randn(rng, 1, nobs)))
        ∇0 = zeros(Float32, 5, nobs); ∇0[5, :] .= 1
        ∇1 = copy(∇0)
        EvoTrees.update_grads!(∇0, p, p[1, :], EvoTrees.GaussianMLE, params, nothing, nothing)
        EvoTrees.update_grads!(∇1, p, p[1, :], EvoTrees.GaussianMLE, params, nothing, DcorCache(c))
        @test Float64.(∇1[1, :]) ≈ nobs .* dcov2_grad(Float64.(p[1, :]), c) .* curv(p[2, :]) rtol = 1e-5
        @test ∇1[2:5, :] == ∇0[2:5, :]
        # two targets: rows 1 and 3 each get their own increment, rows 2 and 4:9 do not move
        p2 = Float32.(vcat(randn(rng, 1, nobs), 0.1f0 .* randn(rng, 1, nobs),
            randn(rng, 1, nobs), 0.1f0 .* randn(rng, 1, nobs)))
        y2 = p2[[1, 3], :]
        ∇a = zeros(Float32, 9, nobs); ∇a[9, :] .= 1
        ∇b = copy(∇a)
        EvoTrees.update_grads!(∇a, p2, y2, EvoTrees.GaussianMLE, params, nothing, nothing)
        EvoTrees.update_grads!(∇b, p2, y2, EvoTrees.GaussianMLE, params, nothing, DcorCache(c))
        for k in (1, 3)
            @test Float64.(∇b[k, :]) ≈ nobs .* dcov2_grad(Float64.(p2[k, :]), c) .* curv(p2[k+1, :]) rtol = 1e-5
        end
        @test ∇b[[2; 4:9], :] == ∇a[[2; 4:9], :]

        # and at the model level the location is decorrelated
        mcfg(λ) = EvoTreeMLE(loss=:gaussian_mle, nrounds=40, max_depth=4, eta=0.2, seed=1, ctrl_lambda=λ)
        mb = fit(mcfg(0.0); x_train=x, y_train=y, verbosity=0)
        mp = fit(mcfg(10.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
        pmp = predict(mp, x)
        @test dcor2(Float64.(pmp[:, 1]), ctrl) < dcor2(Float64.(predict(mb, x)[:, 1]), ctrl) / 2
        @test all(isfinite, pmp) && cor(pmp[:, 1], y) > 0.5
        # :logistic_mle is not supported
        @test_throws "not for logistic_mle" fit(EvoTreeMLE(loss=:logistic_mle, nrounds=5, max_depth=3,
            ctrl_lambda=1.0); x_train=x, y_train=y, ctrl_train=ctrl, verbosity=0)
    end
end

@testset "several controls" begin
    rng = Xoshiro(51)
    nobs = 2_000
    x = randn(rng, nobs, 4)
    # each control tracks one feature; the third feature is unrelated to both
    c1 = x[:, 1] .+ 0.3 .* randn(rng, nobs)
    c2 = x[:, 2] .+ 0.3 .* randn(rng, nobs)
    y = 2 .* x[:, 1] .+ x[:, 2] .+ x[:, 3] .+ 0.2 .* randn(rng, nobs)
    C = hcat(c1, c2)
    gid = repeat(1:40, inner=nobs ÷ 40)
    gi = build_group_index(gid, nobs, "g")
    cfg(λ; within=false) = EvoTreeRegressor(loss=:mse, nrounds=40, max_depth=4, eta=0.2, seed=1,
        ctrl_lambda=λ, ctrl_within_group=within)

    # with y = p, row 1 holds the penalty alone
    p = Float32.(randn(Xoshiro(52), 1, nobs))
    yp = vec(copy(p))
    function penalty(params, ctrl)
        ∇ = zeros(Float32, 3, nobs)
        ∇[3, :] .= 1
        EvoTrees.update_grads!(∇, p, yp, EvoTrees.MSE, params, nothing, ctrl)
        return ∇[1, :]
    end
    b1, b2 = build_ctrl(c1, nobs, "c"), build_ctrl(c2, nobs, "c")
    one_ctrl(params, b) = EvoTrees.dcor_cache(params, b, gi, ones(nobs))
    two_ctrls(params, ω) = EvoTrees.dcor_cache(params, EvoTrees.Controls([b1, b2], ω, ["c1", "c2"]), gi, ones(nobs))

    @testset "one control through the sum is that control exactly" begin
        for within in (false, true)
            params = cfg(3.0; within)
            plain = within ? GroupedDcorCache(b1, gi) : DcorCache(b1)
            @test penalty(params, one_ctrl(params, b1)) == penalty(params, plain)
        end
        # and under a curvature other than that of :mse
        pm = Float32.(vcat(randn(Xoshiro(56), 1, nobs), 0.1f0 .* randn(Xoshiro(57), 1, nobs)))
        function penalty_mle(params, ctrl)
            ∇ = zeros(Float32, 5, nobs)
            ∇[5, :] .= 1
            EvoTrees.update_grads!(∇, pm, pm[1, :], EvoTrees.GaussianMLE, params, nothing, ctrl)
            return ∇[1, :]
        end
        for within in (false, true)
            params = EvoTreeMLE(loss=:gaussian_mle, ctrl_lambda=1.0, ctrl_within_group=within)
            plain = within ? GroupedDcorCache(b1, gi) : DcorCache(b1)
            @test penalty_mle(params, one_ctrl(params, b1)) == penalty_mle(params, plain)
        end
        # and at the model level, however the one control is passed
        m0 = fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=c1, verbosity=0)
        for kw in ((ctrl_train=reshape(c1, :, 1),), (ctrl_train=c1, ctrl_weights=[1.0]))
            @test predict(fit(cfg(5.0); x_train=x, y_train=y, kw..., verbosity=0), x) == predict(m0, x)
        end
    end

    @testset "the penalty is the weighted sum of each control's" begin
        for within in (false, true), ω in ([1.0, 1.0], [0.5, 2.0])
            params = cfg(3.0; within)
            expected = ω[1] .* Float64.(penalty(params, one_ctrl(params, b1))) .+
                       ω[2] .* Float64.(penalty(params, one_ctrl(params, b2)))
            @test Float64.(penalty(params, two_ctrls(params, ω))) ≈ expected rtol = 1e-5
        end
        # multi-target: each output row gets both controls' increments
        P2 = Float32.(randn(Xoshiro(53), 2, nobs))
        params = cfg(3.0)
        function penalty2(ctrl)
            ∇ = zeros(Float32, 5, nobs)
            ∇[5, :] .= 1
            EvoTrees.update_grads!(∇, P2, copy(P2), EvoTrees.MSE, params, nothing, ctrl)
            return Float64.(∇[1:2, :])
        end
        @test penalty2(two_ctrls(params, [1.0, 1.0])) ≈
              penalty2(one_ctrl(params, b1)) .+ penalty2(one_ctrl(params, b2)) rtol = 1e-5
        # the gaussian_mle location: only its row moves, by both increments
        mle = EvoTreeMLE(loss=:gaussian_mle, ctrl_lambda=1.0)
        pm = Float32.(vcat(randn(Xoshiro(54), 1, nobs), 0.1f0 .* randn(Xoshiro(55), 1, nobs)))
        function penalty_mle(ctrl)
            ∇ = zeros(Float32, 5, nobs)
            ∇[5, :] .= 1
            EvoTrees.update_grads!(∇, pm, pm[1, :], EvoTrees.GaussianMLE, mle, nothing, ctrl)
            return ∇
        end
        ∇0 = penalty_mle(nothing)
        ∇b = penalty_mle(two_ctrls(mle, [1.0, 1.0]))
        inc(ctrl) = Float64.(penalty_mle(ctrl)[1, :] .- ∇0[1, :])
        @test Float64.(∇b[1, :] .- ∇0[1, :]) ≈ inc(one_ctrl(mle, b1)) .+ inc(one_ctrl(mle, b2)) rtol = 1e-5
        @test ∇b[2:5, :] == ∇0[2:5, :]
    end

    @testset "a group one control skips gets only the other's term" begin
        # c2 is flat on group 1, both on group 2 and c1 on group 3: each group gets the term of the
        # control it can use
        f1, f2 = copy(c1), copy(c2)
        f2[1:50] .= 0.7
        f1[51:100] .= 0.7
        f2[51:100] .= 0.7
        f1[101:150] .= 0.7
        params = cfg(3.0; within=true)
        a1, a2 = build_ctrl(f1, nobs, "c"), build_ctrl(f2, nobs, "c")
        s = penalty(params, EvoTrees.dcor_cache(params, EvoTrees.Controls([a1, a2], [1.0, 1.0], ["c1", "c2"]),
            gi, ones(nobs)))
        s1 = penalty(params, one_ctrl(params, a1))
        s2 = penalty(params, one_ctrl(params, a2))
        @test s[1:50] == s1[1:50]
        @test all(iszero, s[51:100])
        @test s[101:150] == s2[101:150]
        @test count(!iszero, s[151:end]) > 1750
        # a control no group can use is an error naming it; this one is constant within groups only
        @test_throws "column 2 of `ctrl_train`" fit(cfg(1.0; within=true); x_train=x, y_train=y,
            ctrl_train=hcat(c1, Float64.(gid)), group_train=gid, verbosity=0)
    end

    dep(m, c) = dcor2(Float64.(predict(m, x)[:, 1]), c)
    base = fit(cfg(0.0); x_train=x, y_train=y, verbosity=0)

    @testset "both controls lose their dependence" begin
        mb = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=C, verbosity=0)
        m1 = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=c1, verbosity=0)
        @test dep(mb, c1) < dep(base, c1) / 2
        @test dep(mb, c2) < dep(base, c2) / 2
        # a control left out gains dependence, as the fit shifts to its feature
        @test dep(m1, c2) > dep(base, c2)
        # the third feature is unrelated to both and survives
        @test cor(predict(mb, x)[:, 1], x[:, 3]) > 0.5
        # a lighter weight on the first control leaves it more dependence and the second less
        mw = fit(cfg(10.0); x_train=x, y_train=y, ctrl_train=C, ctrl_weights=[0.1, 1.0], verbosity=0)
        @test dep(mw, c1) > 2 * dep(mb, c1)
        @test dep(mw, c2) < dep(mb, c2)
    end

    @testset "table interface" begin
        df = (f1=x[:, 1], f2=x[:, 2], f3=x[:, 3], f4=x[:, 4], c1=c1, c2=c2, y=y)
        mt = fit(cfg(10.0), df; target_name="y", ctrl_name=["c1", "c2"], verbosity=0)
        # both controls are roles, not features
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test mt.info[:ctrl_names] == [:c1, :c2]
        pt = Float64.(predict(mt, df)[:, 1])
        @test dcor2(pt, c1) < dep(base, c1) / 2
        @test dcor2(pt, c2) < dep(base, c2) / 2
        # weights go through the table path too
        mtw = fit(cfg(10.0), df; target_name="y", ctrl_name=["c1", "c2"], ctrl_weights=[0.1, 1.0], verbosity=0)
        @test dcor2(Float64.(predict(mtw, df)[:, 1]), c1) > 2 * dcor2(pt, c1)
        # a vector of one name is the same control as the name alone
        ms = fit(cfg(10.0), df; target_name="y", ctrl_name="c1", verbosity=0)
        mv = fit(cfg(10.0), df; target_name="y", ctrl_name=["c1"], verbosity=0)
        @test predict(mv, df) == predict(ms, df)
        @test mv.info[:ctrl_names] == [:c1]
    end

    @testset "rejected inputs" begin
        df = (f1=x[:, 1], c1=c1, c2=c2, k=fill(1.0, nobs), y=y)
        @test_throws "more than once" fit(cfg(1.0), df; target_name="y", ctrl_name=["c1", "c1"], verbosity=0)
        @test_throws "is empty" fit(cfg(1.0), df; target_name="y", ctrl_name=String[], verbosity=0)
        # a failing control is named
        @test_throws "column `k` of `ctrl_name` is constant" fit(cfg(1.0), df; target_name="y",
            ctrl_name=["c1", "k"], verbosity=0)
        @test_throws "column 2 of `ctrl_train` is constant" fit(cfg(1.0); x_train=x, y_train=y,
            ctrl_train=hcat(c1, fill(1.0, nobs)), verbosity=0)
        # one row per observation, as `x_train`
        @test_throws "rows but there are" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=C[1:10, :],
            verbosity=0)
        @test_throws "rows but there are" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=permutedims(C),
            verbosity=0)
        # the weights: one positive, finite entry per control, and only with controls
        @test_throws "entries but there are 2 controls" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=C,
            ctrl_weights=[1.0], verbosity=0)
        @test_throws "entries but there are 2 controls" fit(cfg(1.0), df; target_name="y", ctrl_name=["c1", "c2"],
            ctrl_weights=[1.0], verbosity=0)
        for bad in ([1.0, 0.0], [1.0, -1.0], [1.0, Inf], [1.0, NaN])
            @test_throws "positive and finite" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=C,
                ctrl_weights=bad, verbosity=0)
        end
        @test_throws "must be a vector" fit(cfg(1.0); x_train=x, y_train=y, ctrl_train=c1, ctrl_weights=2.0,
            verbosity=0)
        @test_throws "without a control variable" fit(cfg(0.0); x_train=x, y_train=y, ctrl_weights=[1.0],
            verbosity=0)
        @test_throws "without a control variable" fit(cfg(0.0), df; target_name="y", ctrl_weights=[1.0],
            verbosity=0)
    end
end

@testset "penalised eval metric" begin
    rng = Xoshiro(61)
    nobs = 2_000
    x = randn(rng, nobs, 4)
    c1 = x[:, 1] .+ 0.3 .* randn(rng, nobs)
    c2 = x[:, 2] .+ 0.3 .* randn(rng, nobs)
    y = 2 .* x[:, 1] .+ x[:, 2] .+ x[:, 3] .+ 0.2 .* randn(rng, nobs)
    C = hcat(c1, c2)
    gid = repeat(1:40, inner=nobs ÷ 40)
    wv = 0.5 .+ rand(Xoshiro(67), nobs)
    cfg(λ; within=false, kw...) = EvoTreeRegressor(; loss=:mse, nrounds=20, max_depth=4, eta=0.2, seed=1,
        ctrl_lambda=λ, ctrl_within_group=within, kw...)

    @testset "the value from the sweep is dcov2" begin
        n = 300
        c = build_ctrl(c1[1:n], n, "c")
        for xv in (randn(Xoshiro(62), n), round.(randn(Xoshiro(63), n); digits=1), 1e12 .+ randn(Xoshiro(64), n))
            @test EvoTrees.dcov2_value!(DcorCache(c), xv) ≈ dcov2(Float64.(xv), c) rtol = 1e-8
        end
        @test EvoTrees.dcov2_value!(DcorCache(c), fill(0.3, n)) == 0
        xs = randn(Xoshiro(65), 60)
        cs = build_ctrl(c1[1:60], 60, "c")
        @test EvoTrees.dcov2_value!(DcorCache(cs), xs) ≈ dcov2_naive(xs, cs) rtol = 1e-10
    end

    @testset "on the training set it is the training objective" begin
        # training set as eval set: the metric is the weighted mse plus λ sum_j ω_j dcov2_j, each eval
        # group weighted by its share of the eval weight; the first 3 rows form a group too small to keep
        λ, ω = 5.0, [1.0, 0.5]
        geval = vcat(fill(41, 3), gid[4:end])
        kept = [r for r in (findall(==(g), geval) for g in unique(geval)) if length(r) >= 4]
        for within in (false, true)
            kw = within ? (; group_train=gid, group_eval=geval) : (;)
            m = fit(cfg(λ; within); x_train=x, y_train=y, w_train=wv, ctrl_train=C, ctrl_weights=ω, x_eval=x,
                y_eval=y, w_eval=wv, ctrl_eval=C, kw..., verbosity=0)
            lg = m.info[:logger]
            @test lg[:penalized]
            dep(p, c) = within ? sum(sum(wv[r]) / sum(wv) * dcov2(p[r], build_ctrl(c[r], length(r), "c")) for r in kept) :
                        dcov2(p, build_ctrl(c, nobs, "c"))
            for (j, it) in enumerate(lg[:iter])
                p = Float64.(predict(m, x; ntree_limit=it)[:, 1])
                d = [dep(p, c1), dep(p, c2)]
                @test all(isapprox.(lg[:ctrl_dependence][j], d; rtol=1e-6, atol=1e-12))
                @test lg[:base_metrics][j] ≈ sum(wv .* (p .- y) .^ 2) / sum(wv) rtol = 1e-5
                @test lg[:metrics][j] ≈ lg[:base_metrics][j] + λ * sum(ω .* d) rtol = 1e-6
            end
            # the bias alone carries no dependence
            @test lg[:ctrl_dependence][1] == [0.0, 0.0]
        end
        # several outputs are averaged
        Y2 = hcat(y, .-y .+ x[:, 3])
        m = fit(cfg(λ); x_train=x, y_train=Y2, ctrl_train=c1, x_eval=x, y_eval=Y2, ctrl_eval=c1, verbosity=0)
        lg = m.info[:logger]
        P = Float64.(predict(m, x))
        b = build_ctrl(c1, nobs, "c")
        @test lg[:ctrl_dependence][end][1] ≈ (dcov2(P[:, 1], b) + dcov2(P[:, 2], b)) / 2 rtol = 1e-6
        @test lg[:metrics][end] - lg[:base_metrics][end] ≈ λ * (dcov2(P[:, 1], b) + dcov2(P[:, 2], b)) / 2 rtol = 1e-6
        # an offset gives round 0 a dependence
        o = 0.5 .* c1
        m = fit(cfg(λ); x_train=x, y_train=y, offset_train=o, ctrl_train=c1, x_eval=x, y_eval=y, offset_eval=o,
            ctrl_eval=c1, verbosity=0)
        lg = m.info[:logger]
        @test lg[:ctrl_dependence][1][1] ≈ dcov2(Float64.(Float32.(o)), b) rtol = 1e-6
        @test lg[:ctrl_dependence][1][1] > 0.1
        @test lg[:metrics][1] ≈ lg[:base_metrics][1] + λ * lg[:ctrl_dependence][1][1] rtol = 1e-9
    end

    @testset "the eval controls are scaled on the eval set" begin
        # held-out eval controls are standardised on the eval set, so an affine change leaves the
        # dependence unchanged
        tr, ev = 1:1500, 1501:2000
        for within in (false, true)
            kw = within ? (; group_train=gid[tr], group_eval=gid[ev]) : (;)
            fitc(ce) = fit(cfg(5.0; within); x_train=x[tr, :], y_train=y[tr], ctrl_train=c1[tr], x_eval=x[ev, :],
                y_eval=y[ev], ctrl_eval=ce, kw..., verbosity=0)
            ma, mb = fitc(c1[ev]), fitc(3 .* c1[ev] .+ 7)
            la, lb = ma.info[:logger], mb.info[:logger]
            @test all(all(isapprox.(a, b; rtol=1e-9, atol=1e-14)) for (a, b) in zip(la[:ctrl_dependence], lb[:ctrl_dependence]))
            @test la[:metrics] ≈ lb[:metrics] rtol = 1e-9
            groups = [findall(==(g), gid[ev]) for g in unique(gid[ev])]
            for (j, it) in enumerate(la[:iter])
                p = Float64.(predict(ma, x[ev, :]; ntree_limit=it)[:, 1])
                ref = within ? sum(length(r) / length(ev) * dcov2(p[r], build_ctrl(c1[ev][r], length(r), "c")) for r in groups) :
                      dcov2(p, build_ctrl(c1[ev], length(ev), "c"))
                @test la[:ctrl_dependence][j][1] ≈ ref rtol = 1e-6 atol = 1e-12
            end
        end
    end

    @testset "its gradient is the training penalty" begin
        # sum(w) / m times the derivative of the eval term equals the training penalty, m being the
        # metric's multiple of the objective: 2 for :poisson, -1 for :gaussian_mle. Exact for :mse,
        # :logloss and :poisson, at y = exp(p) for :gamma and :tweedie, and at a scale constant over the
        # statistic's rows for :gaussian_mle
        n = 400
        g400 = vcat(fill(0, 3), fill(1, 37), fill(2, 120), repeat(3:8, inner=40))
        cb = build_ctrl(c1[1:n], n, "c")
        gi = build_group_index(g400, n, "g")
        w = 0.5 .+ rand(Xoshiro(68), n)
        pr = 0.5 .* randn(Xoshiro(66), n)
        probe = filter(i -> minimum(abs(pr[i] - pr[j]) for j in 1:n if j != i) > 1e-4, 4:n)[1:8]
        cases = (
            (EvoTrees.MSE, :mse, 1, pr, pr),
            (EvoTrees.LogLoss, :logloss, 1, pr, Float64.(pr .> 0)),
            (EvoTrees.Poisson, :poisson, 2, pr, exp.(pr) .+ 0.3),
            (EvoTrees.Gamma, :gamma, 1, pr, exp.(pr)),
            (EvoTrees.Tweedie, :tweedie, 1, pr, exp.(pr)),
            (EvoTrees.GaussianMLE, :gaussian_mle, -1, pr, pr .+ 0.1),
        )
        for (L, loss, mL, p1, yy) in cases, within in (false, true)
            params = L == EvoTrees.GaussianMLE ?
                     EvoTreeMLE(loss=loss, ctrl_lambda=2.0, ctrl_within_group=within) :
                     EvoTreeRegressor(loss=loss, ctrl_lambda=2.0, ctrl_within_group=within)
            # the default metric is the one that includes the penalty
            feval = EvoTrees.metric_dict[params.metric]
            @test feval === EvoTrees._own_metric(L)
            p = L == EvoTrees.GaussianMLE ? vcat(p1', (within ? 0.1 .* g400 : fill(0.2, n))') : Matrix(p1')
            K = size(p, 1)
            ∇0 = zeros(2K + 1, n)
            ∇0[2K+1, :] .= w
            ∇1 = copy(∇0)
            EvoTrees.update_grads!(∇0, p, yy, L, params, nothing, nothing)
            EvoTrees.update_grads!(∇1, p, yy, L, params, nothing, EvoTrees.dcor_cache(params, cb, gi, w))
            inc = ∇1[1, :] .- ∇0[1, :]
            ec = EvoTrees.eval_ctrl(params, L, K, EvoTrees.Controls([cb], [1.0], ["c"]), within ? gi : nothing, feval, w)
            @test ec.penalized
            term(q) = EvoTrees.eval_penalty!(ec, q)[2]
            # rows with no neighbour within 1e-4, so the step crosses no kink
            h = 1e-7
            for i in (within ? vcat(1, probe) : probe)
                qp = copy(p); qp[1, i] += h
                qm = copy(p); qm[1, i] -= h
                @test isapprox(inc[i], sum(w) / mL * (term(qp) - term(qm)) / (2h); rtol=1e-4,
                    atol=1e-6 * maximum(abs, inc))
            end
            # a group too small to keep carries no penalty, in training or in the eval term
            within && @test inc[1] == 0
        end
        # two :gaussian_mle targets with scales that vary by row: each location's term weighs each
        # group's dcov2 by that group's mean of exp(-2 scale), and the targets average
        p4 = vcat(pr', 0.3 .* randn(Xoshiro(69), 1, n), .-pr', 0.2 .* randn(Xoshiro(70), 1, n))
        for within in (false, true)
            params = EvoTreeMLE(loss=:gaussian_mle, ctrl_lambda=2.0, ctrl_within_group=within)
            ec = EvoTrees.eval_ctrl(params, EvoTrees.GaussianMLE, 4, EvoTrees.Controls([cb], [1.0], ["c"]),
                within ? gi : nothing, EvoTrees.gaussian_mle, ones(n))
            rows = within ? [findall(==(g), g400) for g in 1:8] : [collect(1:n)]
            t(k) = sum(length(r) / n * mean(exp.(-2 .* p4[k+1, r])) * dcov2(p4[k, r], within ? build_ctrl(cb[r], length(r), "c") : cb)
                       for r in rows)
            @test EvoTrees.eval_penalty!(ec, p4)[2] ≈ -2.0 / 2 * (t(1) + t(3)) / 2 rtol = 1e-8
        end
    end

    @testset "every admitted loss adds the penalty to its own metric" begin
        yb = Float64.(y .> 0)
        yp = abs.(y) .+ 0.5
        for (params, yy) in (
            (EvoTreeRegressor(loss=:logloss, nrounds=10, max_depth=3, ctrl_lambda=5.0), yb),
            (EvoTreeRegressor(loss=:poisson, nrounds=10, max_depth=3, ctrl_lambda=5.0), yp),
            (EvoTreeRegressor(loss=:gamma, nrounds=10, max_depth=3, ctrl_lambda=5.0), yp),
            (EvoTreeRegressor(loss=:tweedie, nrounds=10, max_depth=3, ctrl_lambda=5.0), yp),
            (EvoTreeMLE(loss=:gaussian_mle, nrounds=10, max_depth=3, ctrl_lambda=5.0), y),
        )
            lg = fit(params; x_train=x, y_train=yy, ctrl_train=c1, x_eval=x, y_eval=yy, ctrl_eval=c1,
                verbosity=0).info[:logger]
            @test lg[:penalized]
            @test lg[:metrics][end] != lg[:base_metrics][end]
            # the log-likelihood is maximised, so the penalty lowers it
            params isa EvoTreeMLE && @test lg[:metrics][end] < lg[:base_metrics][end]
        end
    end

    @testset "other metrics are reported as they are" begin
        for metric in (:rmse, :mae)
            m = fit(cfg(5.0; metric); x_train=x, y_train=y, ctrl_train=c1, x_eval=x, y_eval=y, ctrl_eval=c1, verbosity=0)
            lg = m.info[:logger]
            @test !lg[:penalized]
            @test lg[:metrics] == lg[:base_metrics]
            @test lg[:ctrl_dependence][end][1] > 0
        end
        # a zero weight logs the dependence but not the penalty
        m0 = fit(cfg(0.0); x_train=x, y_train=y, ctrl_train=c1, x_eval=x, y_eval=y, ctrl_eval=c1, verbosity=0)
        @test !m0.info[:logger][:penalized]
        @test m0.info[:logger][:metrics] == m0.info[:logger][:base_metrics]
        # a fit without controls keeps the logger it had
        mn = fit(cfg(0.0); x_train=x, y_train=y, x_eval=x, y_eval=y, verbosity=0)
        @test !haskey(mn.info[:logger], :base_metrics)
        @test mn.info[:logger][:metrics] == m0.info[:logger][:metrics]
    end

    @testset "early stopping follows the penalised metric" begin
        tr, ev = 1:1500, 1501:2000
        m = fit(cfg(20.0; nrounds=300, eta=0.1, early_stopping_rounds=10); x_train=x[tr, :], y_train=y[tr],
            ctrl_train=c1[tr], x_eval=x[ev, :], y_eval=y[ev], ctrl_eval=c1[ev], verbosity=0)
        lg = m.info[:logger]
        @test lg[:penalized]
        @test 0 < lg[:best_iter]
        @test lg[:nrounds] == lg[:best_iter] + 10 < 300
        @test lg[:best_iter] == lg[:iter][argmin(lg[:metrics])]
        # the base metric alone would have run longer
        @test lg[:iter][argmin(lg[:base_metrics])] > lg[:best_iter]
    end

    @testset "table interface" begin
        df = (f1=x[:, 1], f2=x[:, 2], f3=x[:, 3], f4=x[:, 4], c1=c1, c2=c2, y=y)
        mt = fit(cfg(5.0), df; target_name="y", ctrl_name=["c1", "c2"], ctrl_weights=[1.0, 0.5], deval=df, verbosity=0)
        mm = fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=C, ctrl_weights=[1.0, 0.5], x_eval=x, y_eval=y,
            ctrl_eval=C, verbosity=0)
        lt, lm = mt.info[:logger], mm.info[:logger]
        @test lt[:penalized]
        @test lt[:metrics] == lm[:metrics]
        @test lt[:ctrl_dependence] == lm[:ctrl_dependence]
        @test_throws "has no column `c2`" fit(cfg(5.0), df; target_name="y", ctrl_name=["c1", "c2"],
            deval=(f1=x[:, 1], f2=x[:, 2], f3=x[:, 3], f4=x[:, 4], c1=c1, y=y), verbosity=0)
        @test_throws "column `c1` of `deval` is constant, so its dependence on the evaluation set is undefined" fit(
            cfg(5.0), df; target_name="y", ctrl_name="c1", deval=merge(df, (c1=fill(1.0, nobs),)), verbosity=0)
        # within groups the eval set needs its groups
        dg = merge(df, (g=gid,))
        @test fit(cfg(5.0; within=true), dg; target_name="y", ctrl_name="c1", group_name="g", deval=dg,
            verbosity=0).info[:logger][:penalized]
        @test_throws "eval set has no groups" fit(cfg(5.0; within=true), dg; target_name="y", ctrl_name="c1",
            group_name="g", eval_group_name=nothing, deval=dg, verbosity=0)
    end

    @testset "rejected inputs" begin
        @test_throws "not for the eval set" fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=c1, x_eval=x, y_eval=y,
            verbosity=0)
        @test_throws "without `ctrl_train`" fit(cfg(0.0); x_train=x, y_train=y, x_eval=x, y_eval=y, ctrl_eval=c1,
            verbosity=0)
        @test_throws "controls but `ctrl_train` has" fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=C, x_eval=x,
            y_eval=y, ctrl_eval=c1, verbosity=0)
        @test_throws "`ctrl_eval` has 10 rows but there are 2000" fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=C,
            x_eval=x, y_eval=y, ctrl_eval=C[1:10, :], verbosity=0)
        @test_throws "`ctrl_eval` is constant" fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=c1, x_eval=x,
            y_eval=y, ctrl_eval=fill(1.0, nobs), verbosity=0)
        @test_throws "column 2 of `ctrl_eval` is constant" fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=C, x_eval=x,
            y_eval=y, ctrl_eval=hcat(c1, fill(1.0, nobs)), verbosity=0)
        @test_throws "eval set has no groups" fit(cfg(5.0; within=true); x_train=x, y_train=y, ctrl_train=c1,
            group_train=gid, x_eval=x, y_eval=y, ctrl_eval=c1, verbosity=0)
        # eval controls without an eval set warn as `x_eval` alone does
        @test_logs (:warn, r"both `x_eval` and `y_eval`") fit(cfg(5.0); x_train=x, y_train=y, ctrl_train=c1,
            ctrl_eval=c1, verbosity=0)
    end
end

@testset "loss = :pearson" begin
    zscore(v) = (v .- mean(v)) ./ std(v)
    # 100 groups of 50 rows, shuffled. Within a group the control is the row-level term `dev`,
    # which also carries a third of the target's signal within the group
    function panel(seed)
        rng = Xoshiro(seed)
        nobs = 100 * 50
        group = repeat(1:100, inner=50)
        level = randn(rng, 100)[group]
        dev = randn(rng, nobs)
        x = hcat(dev .+ 0.3 .* randn(rng, nobs), randn(rng, nobs, 3), level .+ 0.1 .* randn(rng, nobs))
        y = 3 .* level .+ dev .+ x[:, 2] .+ 0.5 .* randn(rng, nobs)
        perm = randperm(rng, nobs)
        return (; group=group[perm], x=x[perm, :], ctrl=(level.+dev)[perm], y=y[perm])
    end

    @testset "within groups only" begin
        P = panel(1)
        cfg(within) = EvoTreeRegressor(loss=:pearson, nrounds=5, max_depth=3, ctrl_lambda=1.0, ctrl_within_group=within)
        @test_throws "Set `ctrl_within_group = true`" fit(cfg(false); x_train=P.x, y_train=P.y, group_train=P.group,
            ctrl_train=P.ctrl, verbosity=0)
        @test fit(cfg(true); x_train=P.x, y_train=P.y, group_train=P.group, ctrl_train=P.ctrl,
            verbosity=0) isa EvoTrees.EvoTree
    end

    @testset "the penalty lands on the gradient unchanged" begin
        # the loss's curvature is 2w, so each kept group adds `λ * W_g` times its gradient, `W_g` its weight
        # sum. Group 3 is too small for the penalty, and group 1 has a constant target, so the loss gives it
        # no curvature and no penalty. Over 21 seeds the error was at most 1.1e-7, Float32 rounding
        rng = Xoshiro(91)
        gid = vcat([fill(g, m) for (g, m) in enumerate((9, 12, 3, 10, 8))]...)
        gid = gid[randperm(rng, length(gid))]
        n = length(gid)
        ctrl = randn(rng, n)
        y = Float32.(3 .* randn(rng, n) .+ 10)
        y[gid.==1] .= 7.0f0
        p = Float32.(0.4 .* randn(rng, 1, n))
        w = Float32.(0.5 .+ rand(rng, n))
        gi = build_group_index(gid, n, "g")
        params = EvoTreeRegressor(loss=:pearson, ctrl_lambda=2.0, ctrl_within_group=true)
        ∇a = zeros(Float32, 3, n)
        ∇a[3, :] .= w
        ∇b = copy(∇a)
        EvoTrees.update_grads!(∇a, p, y, EvoTrees.Pearson, params, gi, nothing)
        EvoTrees.update_grads!(∇b, p, y, EvoTrees.Pearson, params, gi, EvoTrees.dcor_cache(params, ctrl, gi, w))
        expected = zeros(n)
        for g in (2, 4, 5)
            r = findall(==(g), gid)
            expected[r] .= 2.0 * sum(Float64.(w[r])) .* dcov2_grad(Float64.(p[1, r]), zscore(ctrl[r]))
        end
        @test Float64.(∇b[1, :] .- ∇a[1, :]) ≈ expected rtol = 1e-5
        skipped = (gid .== 1) .| (gid .== 3)
        @test ∇b[1, skipped] == ∇a[1, skipped]
        @test ∇b[2:3, :] == ∇a[2:3, :]
    end

    @testset "metric = :pearson_mse is the training objective" begin
        # the loss's weighted squared error per unit weight, so its gradient times the total weight is the
        # loss's gradient row, and with the penalty the penalised eval value's is the training row. A group
        # of one row and one with a constant target are not scored. Over 21 seeds: value 6.0e-16
        # relative, gradients within 2.4e-10 and 8.8e-10 of the largest row
        rng = Xoshiro(96)
        gid = repeat(1:8, inner=40)
        gid[1] = 9
        n = length(gid)
        gi = build_group_index(gid, n, "g")
        c1, c2 = randn(rng, n), randn(rng, n)
        y = c1 .+ randn(rng, n) .+ 5 .* gid
        y[gid.==2] .= 0.5
        w = 0.5 .+ rand(rng, n)
        p = 0.5 .* randn(rng, 1, n)
        metric(q) = EvoTrees.pearson_mse(q, y, w, similar(w); group=gi)
        obj = 0.0
        for g in (1, 3, 4, 5, 6, 7, 8)
            r = findall(==(g), gid)
            sw = sum(w[r])
            pb, yb = sum(w[r] .* p[1, r]) / sw, sum(w[r] .* y[r]) / sw
            sy = sqrt(sum(w[r] .* (y[r] .- yb) .^ 2) / sw)
            obj += sum(w[r] .* ((p[1, r] .- pb) .- (y[r] .- yb) ./ sy) .^ 2) / sum(w)
        end
        @test metric(p) ≈ obj rtol = 1e-12
        fd(f, i, h) = (qp = copy(p); qp[1, i] += h; qm = copy(p); qm[1, i] -= h; sum(w) * (f(qp) - f(qm)) / (2h))
        ∇0 = zeros(3, n)
        ∇0[3, :] .= w
        EvoTrees.update_grads!(∇0, p, y, EvoTrees.Pearson, EvoTreeRegressor(loss=:pearson), gi)
        @test maximum(abs, [fd(metric, i, 1e-4) for i in 1:n] .- ∇0[1, :]) < 1e-7 * maximum(abs, ∇0[1, :])
        ctrls = EvoTrees.Controls([build_ctrl(c1, n, "c"), build_ctrl(c2, n, "c")], [1.0, 0.5], ["c1", "c2"])
        params = EvoTreeRegressor(loss=:pearson, metric=:pearson_mse, ctrl_lambda=2.0, ctrl_within_group=true)
        ec = EvoTrees.eval_ctrl(params, EvoTrees.Pearson, 1, ctrls, gi, EvoTrees.pearson_mse, w)
        @test ec.penalized
        ∇ = zeros(3, n)
        ∇[3, :] .= w
        EvoTrees.update_grads!(∇, p, y, EvoTrees.Pearson, params, gi, EvoTrees.dcor_cache(params, ctrls, gi, w))
        value(q) = metric(q) + EvoTrees.eval_penalty!(ec, q)[2]
        # rows with no neighbour within 1e-4 in their group, so the step crosses no kink of the statistic
        probe = filter(i -> 2 < gid[i] < 9 &&
            minimum(abs(p[i] - p[j]) for j in 1:n if j != i && gid[j] == gid[i]) > 1e-4, 1:n)[1:8]
        @test maximum(abs(∇[1, i] - fd(value, i, 1e-5)) for i in probe) < 1e-6 * maximum(abs, ∇[1, :])
    end

    @testset "early stopping" begin
        # the default metric, the per-group correlation, is reported as it stands. The first tree sees no
        # penalty and the correlation falls as the penalty sheds the control's share of the signal, so
        # over 21 seeds it stopped at round 1 every time; `:pearson_mse` ran 26 to 62 rounds, to a
        # within-group dcor2 at most 0.11 of round 1's
        P = panel(2)
        tr, ev = P.group .<= 70, P.group .> 70
        rows = [findall(==(d), P.group[ev]) for d in 71:100]
        dep(m, it) = (q = Float64.(predict(m, P.x[ev, :]; ntree_limit=it)[:, 1]);
                      mean(dcor2(q[r], P.ctrl[ev][r]) for r in rows))
        fitc(loss, metric) = fit(EvoTreeRegressor(; loss, metric, nrounds=200, max_depth=4, eta=0.1, seed=1,
                early_stopping_rounds=10, ctrl_lambda=10.0, ctrl_within_group=true);
            x_train=P.x[tr, :], y_train=P.y[tr], group_train=P.group[tr], ctrl_train=P.ctrl[tr], x_eval=P.x[ev, :],
            y_eval=P.y[ev], group_eval=P.group[ev], ctrl_eval=P.ctrl[ev], verbosity=0)
        ma, mb = fitc(:pearson, nothing), fitc(:pearson, :pearson_mse)
        la, lb = ma.info[:logger], mb.info[:logger]
        @test !la[:penalized]
        @test la[:metrics] == la[:base_metrics]
        @test la[:best_iter] == 1
        @test lb[:penalized]
        @test lb[:metrics] ≈ lb[:base_metrics] .+ 10.0 .* first.(lb[:ctrl_dependence]) rtol = 1e-9
        @test lb[:best_iter] == lb[:iter][argmin(lb[:metrics])]
        @test lb[:nrounds] == lb[:best_iter] + 10 < 200
        @test dep(mb, lb[:best_iter]) < dep(ma, la[:best_iter]) / 2
        # under another loss it is reported as it stands, and it needs the eval groups
        lm = fitc(:mse, :pearson_mse).info[:logger]
        @test !lm[:penalized] && lm[:metrics] == lm[:base_metrics]
        @test_throws "`metric = :pearson_mse` requires group information" fit(
            EvoTreeRegressor(loss=:mse, metric=:pearson_mse, nrounds=2); x_train=P.x[tr, :], y_train=P.y[tr],
            x_eval=P.x[ev, :], y_eval=P.y[ev], verbosity=0)
    end
end
