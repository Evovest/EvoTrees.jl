using Test
using Statistics
using Random
using EvoTrees
using EvoTrees: fit, predict, importance, build_group_index, ngroups, group_rows, pearson, _pearson_group

# Margins and tolerances of the fit tests, set from 20 seeds of each test's panel: a tolerance at
# least 100x above the largest error observed, a margin at most a third of the smallest gap.
const F2B_RTOL = 1e-5                 # F2b round-1 identity, unequal sizes and weights (norm isapprox; equal up to Float32 rounding of g)
const F3_AFFINE_METRIC_ATOL = 2e-6    # F3 per-date affine target, eval metric agreement
const F4B_MAX_SHARE = 0.25            # F4b gain share of a date-constant feature at depth 4
const F5_PLATEAU_MARGIN = 0.09        # F5 best in-sample metric over the iter-1 metric
const F5_MSE_MARGIN = 0.0025          # F5 best eval metric over raw :mse's best eval metric
const F7_RTOL = 1e-6                  # F7 table and matrix APIs (norm isapprox; expected equal)
const F9_RTOL = 3e-5                  # F9 continuation from an offset (norm isapprox)
const RK2_PEARSON_MIN_CHANGE = 1e-3   # RK2 relative norm by which :pearson must move

# `isapprox` on arrays compares norms, `norm(a - b) <= rtol * max(norm(a), norm(b))`, so a
# tolerance on it bounds the whole vector, not each entry. It is used where some entries can be
# near zero (U6 g, U9, F2, F3, F7, F9, RK2); elsewhere entries are compared one by one, or
# through a maximum over entries. The unit tolerances are at least 100x the Float64 rounding
# error estimated for each check.

# Weights of the constant-target date in the unit panel. With these, the weighted Float64
# variance of the constant 0.1f0 target, summed in row order, comes out above zero, so a
# variance test would score the date.
const _PL_CONST_W = Float32[1.33, 1.81, 1.6, 0.61, 0.74, 1.77, 0.21, 1.68, 1.63, 1.04, 0.75, 0.7]
const _PL_FEATS = ["x1", "x2", "x3", "x4", "dconst"]

# Five dates of 1, 7, 12, 30 and 45 rows in scattered row order. Date 3 has a constant Float32
# target, date 1 a single row. Target levels and scales differ widely between dates.
function _pl_unit_panel(seed=20260929)
    rng = Xoshiro(seed)
    sizes = [1, 7, 12, 30, 45]
    D = length(sizes)
    date = reduce(vcat, [fill(d, s) for (d, s) in enumerate(sizes)])
    date = date[randperm(rng, length(date))]
    N = length(date)
    w = Float64.(Float32.(0.2 .+ 1.8 .* rand(rng, N)))
    lvl = -50 .+ 100 .* rand(rng, D)
    scl = 10 .^ (-2 .+ 3.7 .* rand(rng, D))
    y = Float32.(lvl[date] .+ scl[date] .* randn(rng, N))
    c3 = findall(==(3), date)
    y[c3] .= 0.1f0
    w[c3] .= Float64.(_PL_CONST_W)
    p = 5 .+ (6 .* rand(rng, D) .- 3)[date] .+ 0.3 .* randn(rng, N)
    return (; date, N, D, sizes, w, y, p, scl, gi=build_group_index(date))
end

# Per-date moments in Float64, scored as the loss scores a date.
function _pl_stats(p::AbstractVector, y, w, gi)
    ng = ngroups(gi)
    scored = falses(ng)
    n = zeros(Int, ng)
    sw, pbar, ybar, sy = zeros(ng), zeros(ng), zeros(ng), ones(ng)
    for g in 1:ng
        rows = group_rows(gi, g)
        n[g] = length(rows)
        sw[g] = sum(Float64(w[r]) for r in rows)
        pbar[g] = sum(w[r] * p[r] for r in rows) / sw[g]
        ybar[g] = sum(w[r] * y[r] for r in rows) / sw[g]
        scored[g] = n[g] >= 2 && minimum(y[rows]) != maximum(y[rows])
        scored[g] && (sy[g] = sqrt(sum(w[r] * (y[r] - ybar[g])^2 for r in rows) / sw[g]))
    end
    return (; scored, n, sw, pbar, ybar, sy)
end

# L(p) = sum over scored dates of sum_i w_i (p_i - pbar_d - yt_i)^2: each row counts once, so a
# date weighs by its size.
function _pl_objective(p::AbstractVector, y, w, gi)
    s = _pl_stats(p, y, w, gi)
    L = 0.0
    for g in 1:ngroups(gi)
        s.scored[g] || continue
        for r in group_rows(gi, g)
            L += w[r] * ((p[r] - s.pbar[g]) - (y[r] - s.ybar[g]) / s.sy[g])^2
        end
    end
    return L
end

function _pl_grads(p::AbstractVector, y, w, gi; T=Float64)
    N = length(p)
    ∇ = zeros(T, 3, N)
    ∇[3, :] .= w
    EvoTrees._pearson_grads!(∇, reshape(T.(p), 1, N), y, gi)
    return ∇
end

# The per-date standardised target, summed in row order within a date as the loss sums it.
function _pl_ytilde(y32::AbstractVector{Float32}, w32::AbstractVector{Float32}, gi)
    yt = zeros(Float64, length(y32))
    for g in 1:ngroups(gi)
        rows = group_rows(gi, g)
        sw = 0.0
        swy = 0.0
        for r in rows
            wr = Float64(w32[r])
            sw += wr
            swy += wr * y32[r]
        end
        ybar = swy / sw
        vy = 0.0
        for r in rows
            d = y32[r] - ybar
            vy += w32[r] * d * d
        end
        sy = sqrt(vy / sw)
        for r in rows
            yt[r] = (y32[r] - ybar) / sy
        end
    end
    return yt
end

_pl_err(f) =
    try
        f()
        nothing
    catch e
        e
    end
_pl_msg(e) = e isa ErrorException ? e.msg : ""

# 60 dates of 80 to 300 assets, rows grouped by date. y = level_d + vol_d (0.3 f(x) + noise),
# with level shocks far larger than the spread within a date and vol varying 10x across dates.
# `dconst` is constant within a date and tracks level_d. Dates 1 to 45 train, 46 to 60 evaluate.
function _pl_fit_panel(seed; sizes=nothing, heavy=false)
    rng = Xoshiro(seed)
    ndates = 60
    sizes = isnothing(sizes) ? rand(rng, 80:300, ndates) : sizes
    date = reduce(vcat, [fill(d, sizes[d]) for d in 1:ndates])
    N = length(date)
    level = 20 .* randn(rng, ndates)
    vol = 10 .^ (rand(rng, ndates) .- 0.5)
    x = randn(rng, N, 4)
    f = x[:, 1] .- 0.5 .* x[:, 2] .+ 0.5 .* x[:, 1] .* x[:, 3]
    f ./= std(f)
    # Student-t with 3 degrees of freedom, scaled to unit variance
    e = heavy ? randn(rng, N) ./ sqrt.(sum(randn(rng, N, 3) .^ 2; dims=2)[:, 1] ./ 3) ./ sqrt(3) :
        randn(rng, N)
    y = level[date] .+ vol[date] .* (0.3 .* f .+ e)
    dconst = (level .+ 0.1 .* randn(rng, ndates))[date]
    tr = date .<= 45
    return (; x=hcat(x, dconst), y, date, tr, ev=.!tr, level, vol, sizes)
end

_pl_fit(cfg, P; kw...) = fit(cfg; x_train=P.x[P.tr, :], y_train=P.y[P.tr], group_train=P.date[P.tr],
    feature_names=_PL_FEATS, verbosity=0, kw...)
_pl_fit_eval(cfg, P; kw...) = _pl_fit(cfg, P; x_eval=P.x[P.ev, :], y_eval=P.y[P.ev],
    group_eval=P.date[P.ev], kw...)

@testset "pearson loss" begin

    @testset "U1 gradient against finite differences" begin
        # Catches post-multiplied weights, a missing factor of 2, unweighted moments and any
        # date factor. The objective is quadratic, so only rounding remains.
        U = _pl_unit_panel()
        g = _pl_grads(U.p, U.y, U.w, U.gi)[1, :]
        h = 1e-6 * maximum(abs, U.p)
        fd = map(1:U.N) do i
            pp, pm = copy(U.p), copy(U.p)
            pp[i] += h
            pm[i] -= h
            (_pl_objective(pp, U.y, U.w, U.gi) - _pl_objective(pm, U.y, U.w, U.gi)) / (2h)
        end
        @test maximum(abs, g .- fd) / maximum(abs, fd) < 1e-6
    end

    @testset "U2 curvature and centring" begin
        # Catches an exact or clipped Pearson Hessian, a date factor, and a target not centred
        # per date.
        U = _pl_unit_panel()
        ∇ = _pl_grads(U.p, U.y, U.w, U.gi)
        s = _pl_stats(U.p, U.y, U.w, U.gi)
        for g in 1:ngroups(U.gi)
            s.scored[g] || continue
            rows = group_rows(U.gi, g)
            # elementwise: h is never near zero
            @test all(isapprox.(∇[2, rows], 2 .* U.w[rows]; rtol=1e-12))
            @test abs(sum(∇[1, rows])) / sum(abs, ∇[1, rows]) < 1e-10
        end
    end

    @testset "U3 per-date shift of the prediction" begin
        # Catches a prediction that is not centred per date.
        U = _pl_unit_panel()
        rng = Xoshiro(3)
        g0 = _pl_grads(U.p, U.y, U.w, U.gi)[1, :]
        shift = (200 .* rand(rng, U.D) .- 100)[U.date]
        g1 = _pl_grads(U.p .+ shift, U.y, U.w, U.gi)[1, :]
        @test maximum(abs, g1 .- g0) < 1e-10 * maximum(abs, g0)
    end

    @testset "U4 per-date affine target" begin
        # Catches centring without scaling, and standardising over all dates rather than per date.
        # The shift is drawn relative to each date's scale so that rounding of the transformed
        # Float64 target stays far below the tolerance.
        U = _pl_unit_panel()
        rng = Xoshiro(4)
        α = 10 .^ (2 .* rand(rng, U.D) .- 1)
        β = α .* U.scl .* (200 .* rand(rng, U.D) .- 100)
        y64 = Float64.(U.y)
        ya = α[U.date] .* y64 .+ β[U.date]
        g0 = _pl_grads(U.p, y64, U.w, U.gi)[1, :]
        g1 = _pl_grads(U.p, ya, U.w, U.gi)[1, :]
        @test maximum(abs, g1 .- g0) < 1e-9 * maximum(abs, g0)
    end

    @testset "U5 identity with the metric's correlation" begin
        # Per date, sum_i a_i (dp_i - yt_i)^2 = (sp - r)^2 + 1 - r^2 with r the metric's own value.
        U = _pl_unit_panel()
        s = _pl_stats(U.p, U.y, U.w, U.gi)
        P = reshape(U.p, 1, :)
        rhs = 0.0
        for g in 1:ngroups(U.gi)
            s.scored[g] || continue
            rows = group_rows(U.gi, g)
            r = _pearson_group(P, U.y, U.w, rows, 1)
            sp = sqrt(sum(U.w[i] * (U.p[i] - s.pbar[g])^2 for i in rows) / s.sw[g])
            rhs += s.sw[g] * ((sp - r)^2 + 1 - r^2)
        end
        @test _pl_objective(U.p, U.y, U.w, U.gi) ≈ rhs rtol = 1e-10
        # The shipped gradient carries the same objective: on a scored row g = 2 w e and h = 2 w,
        # so g^2 / 2h = w e^2, which sums over scored rows to L.
        ∇ = _pl_grads(U.p, U.y, U.w, U.gi)
        Lg = sum(∇[1, r]^2 / (2 * ∇[2, r]) for r in 1:U.N if ∇[2, r] > 0)
        @test Lg ≈ rhs rtol = 1e-10
    end

    @testset "U6 flat date" begin
        # A date with equal predictions gets the round-1 gradient of :mse on its standardised
        # target, with no special rule.
        U = _pl_unit_panel()
        pf = copy(U.p)
        rows4 = findall(==(4), U.date)
        pf[rows4] .= 7.25
        ∇ = _pl_grads(pf, U.y, U.w, U.gi)
        s = _pl_stats(pf, U.y, U.w, U.gi)
        yt = (U.y[rows4] .- s.ybar[4]) ./ s.sy[4]
        # norm isapprox: entries of g near zero are compared through the date's norm
        @test ∇[1, rows4] ≈ -2 .* U.w[rows4] .* yt rtol = 1e-12
        @test all(isapprox.(∇[2, rows4], 2 .* U.w[rows4]; rtol=1e-12))
    end

    @testset "U7 degenerate dates" begin
        # Catches a variance test for a constant target (the constant date's Float64 variance is
        # above zero), values left over from a previous round, and NaN from 0/0.
        U = _pl_unit_panel()
        ∇ = fill(NaN, 3, U.N)
        ∇[3, :] .= U.w
        EvoTrees._pearson_grads!(∇, reshape(U.p, 1, :), U.y, U.gi)
        dead = findall(in((1, 3)), U.date)
        @test length(dead) == 13
        @test all(==(0.0), ∇[1:2, dead])
        @test all(isfinite, ∇)
        @test all(>(0), ∇[2, setdiff(1:U.N, dead)])
    end

    @testset "U8 errors" begin
        # A panel with no date to correlate: one single-row date and one constant date.
        gi = build_group_index([1, 2, 2, 2])
        ∇ = zeros(3, 4)
        ∇[3, :] .= 1
        e = _pl_err(() -> EvoTrees._pearson_grads!(∇, zeros(1, 4), Float32[1, 2, 2, 2], gi))
        @test e isa ErrorException
        @test occursin("`loss = :pearson`", _pl_msg(e)) && occursin("nothing to correlate", _pl_msg(e))

        params = EvoTreeRegressor(; loss=:pearson, nrounds=1)
        e = _pl_err(() -> EvoTrees.update_grads!(zeros(3, 4), zeros(1, 4), Float32[1, 2, 3, 4],
            EvoTrees.Pearson, params, nothing))
        @test e isa ErrorException
        @test occursin(r"group_name", _pl_msg(e))
    end

    @testset "U9 weights stay within their date" begin
        # Catches a global weight normalisation, and a date factor built from weight sums, such
        # as the mean date weight over the date's own. Scaling by 4 is exact in binary, so the
        # other rows must not move at all.
        U = _pl_unit_panel()
        ∇0 = _pl_grads(U.p, U.y, U.w, U.gi)
        w4 = copy(U.w)
        rows4 = findall(==(4), U.date)
        w4[rows4] .*= 4
        ∇1 = _pl_grads(U.p, U.y, w4, U.gi)
        # norm isapprox over the date
        @test ∇1[1, rows4] ≈ 4 .* ∇0[1, rows4] rtol = 1e-12
        @test ∇1[2, rows4] ≈ 4 .* ∇0[2, rows4] rtol = 1e-12
        others = setdiff(1:U.N, rows4)
        @test ∇1[1:2, others] == ∇0[1:2, others]
    end

    @testset "U10 independent of the thread partition" begin
        # A serial pass over the per-date helpers, in date order, must give the same bits.
        U = _pl_unit_panel()
        ∇ = _pl_grads(U.p, U.y, U.w, U.gi)
        ref = zeros(3, U.N)
        ref[3, :] .= U.w
        P = reshape(U.p, 1, :)
        ng = ngroups(U.gi)
        stats = zeros(3, ng)
        scored = zeros(Bool, ng)
        for g in 1:ng
            EvoTrees._pearson_stats_chunk!(stats, scored, ref, P, U.y, U.gi, g:g)
        end
        for g in 1:ng
            EvoTrees._pearson_write_chunk!(ref, P, U.y, stats, scored, U.gi, g:g)
        end
        @test ∇ == ref
    end

    @testset "U11 round 1 is :mse on the standardised target" begin
        # At p = 0, g = 2 w (0 - yt) and h = 2 w on every scored date, whatever its size: the
        # row weight is the :mse weight, with no date factor. Pins this without fitting.
        U = _pl_unit_panel()
        ∇ = _pl_grads(zeros(U.N), U.y, U.w, U.gi)
        ok = falses(U.D)
        for g in 1:U.D
            rows = group_rows(U.gi, g)
            ok[g] = length(rows) >= 2 && minimum(U.y[rows]) != maximum(U.y[rows])
        end
        gref = zeros(U.N)
        href = zeros(U.N)
        for g in 1:U.D
            ok[g] || continue
            rows = group_rows(U.gi, g)
            sw, swy = 0.0, 0.0
            for r in rows
                sw += U.w[r]
                swy += U.w[r] * U.y[r]
            end
            ybar = swy / sw
            vy = 0.0
            for r in rows
                d = U.y[r] - ybar
                vy += U.w[r] * d * d
            end
            sy = sqrt(vy / sw)
            for r in rows
                gref[r] = 2 * U.w[r] * (0 - (U.y[r] - ybar) / sy)
                href[r] = 2 * U.w[r]
            end
        end
        @test ∇[1, :] == gref
        @test ∇[2, :] == href
    end

    @testset "U12 a tiny date carries a row's curvature, no more" begin
        # Dates of 2 to 5 rows beside dates of 120 to 200. Every row gets h = 2 w exactly, so a
        # row of a tiny date carries the curvature of any other row with the same weight, and
        # the tiny dates' share of total curvature is their share of total weight. Catches a
        # date factor nbar / n_d, under which row t of the 3-row date would carry 50 times the
        # curvature of row b of the 150-row date (nbar = 80) at the same weight. Holds for both
        # losses, which share the gradient.
        rng = Xoshiro(12)
        sizes = [3, 2, 5, 150, 200, 120]
        date = reduce(vcat, [fill(d, s) for (d, s) in enumerate(sizes)])
        date = date[randperm(rng, length(date))]
        N = length(date)
        w = Float64.(Float32.(0.2 .+ 1.8 .* rand(rng, N)))
        tiny = findall(<=(3), date)
        t, b = findfirst(==(1), date), findfirst(==(4), date)
        w[t] = w[b]
        y = randn(rng, N)
        p = reshape(randn(rng, N), 1, N)
        gi = build_group_index(date)
        params = EvoTreeRegressor(; loss=:pearson)
        for L in (EvoTrees.Pearson,)
            ∇ = zeros(3, N)
            ∇[3, :] .= w
            EvoTrees.update_grads!(∇, p, y, L, params, gi)
            @test ∇[2, :] == 2 .* w
            @test ∇[2, t] == ∇[2, b]
            @test sum(∇[2, tiny]) / sum(∇[2, :]) ≈ sum(w[tiny]) / sum(w) rtol = 1e-12
            # a tiny date alone gives the same bits: nothing from the other dates enters it
            for d in 1:3
                rows = findall(==(d), date)
                ∇d = zeros(3, length(rows))
                ∇d[3, :] .= w[rows]
                EvoTrees.update_grads!(∇d, p[:, rows], y[rows], L, params, build_group_index(date[rows]))
                @test ∇d == ∇[:, rows]
            end
        end
    end

    @testset "F1 defaults" begin
        @test EvoTreeRegressor(; loss=:pearson).metric == :pearson
        P = _pl_fit_panel(11)
        m = _pl_fit(EvoTreeRegressor(; loss=:pearson, nrounds=5, max_depth=3), P)
        @test m.bias == [0f0]
        @test all(isfinite, predict(m, P.x[P.ev, :]))
    end

    @testset "F2a round-1 identity, equal sizes and unit weights" begin
        # With every date the same size and unit weights, the first tree must be the :mse tree
        # on the per-date standardised target. Catches flat-date rules, a fitted bias and a
        # factor of 2. Expected bitwise; the norm isapprox leaves room for rounding only.
        P = _pl_fit_panel(12; sizes=fill(150, 60))
        xtr, ytr, dtr = P.x[P.tr, :], P.y[P.tr], P.date[P.tr]
        n = length(ytr)
        gi = build_group_index(dtr)
        yt = _pl_ytilde(Float32.(ytr), ones(Float32, n), gi)
        cfg(loss) = EvoTreeRegressor(; loss, nrounds=1, max_depth=4, eta=0.1)
        mp = fit(cfg(:pearson); x_train=xtr, y_train=ytr, group_train=dtr, verbosity=0)
        mm = fit(cfg(:mse); x_train=xtr, y_train=yt, group_train=dtr, offset_train=zeros(n), verbosity=0)
        @test predict(mp, xtr) ≈ predict(mm, xtr) rtol = 1e-6
    end

    @testset "F2b round-1 identity, unequal sizes and weights" begin
        # Dates of 80 to 300 rows and random weights. Each row counts once, so the :mse fit on
        # the standardised target takes the same weights and the same settings, and the first
        # trees must agree as in F2a. Catches a date factor, which only unequal sizes expose,
        # and weights applied twice or not at all. With weights other than 1, :mse rounds g
        # twice in Float32 (2 (p - yt), then times w) where the loss rounds 2 w yt once, so g
        # can differ in the last bit; the norm isapprox leaves room for that rounding only.
        P = _pl_fit_panel(13)
        rng = Xoshiro(13)
        xtr, ytr, dtr = P.x[P.tr, :], P.y[P.tr], P.date[P.tr]
        n = length(ytr)
        w = Float32.(0.2 .+ 1.8 .* rand(rng, n))
        gi = build_group_index(dtr)
        yt = _pl_ytilde(Float32.(ytr), w, gi)
        cfg(loss) = EvoTreeRegressor(; loss, nrounds=1, max_depth=4, eta=0.1)
        mp = fit(cfg(:pearson); x_train=xtr, y_train=ytr, w_train=w, group_train=dtr, verbosity=0)
        mm = fit(cfg(:mse); x_train=xtr, y_train=yt, w_train=w, group_train=dtr,
            offset_train=zeros(n), verbosity=0)
        # norm isapprox
        @test predict(mp, xtr) ≈ predict(mm, xtr) rtol = F2B_RTOL
    end

    @testset "F3 per-date target scale" begin
        # Scaling a Float32 target by a power of two leaves the standardised target bitwise
        # unchanged, so the model must not move. Catches global standardisation.
        P = _pl_fit_panel(14)
        rng = Xoshiro(14)
        k = rand(rng, -3:3, 60)
        y2 = P.y .* 2.0 .^ k[P.date]
        P2 = merge(P, (; y=y2))
        cfg(loss) = EvoTreeRegressor(; loss, nrounds=20, max_depth=4, eta=0.1)
        xev = P.x[P.ev, :]
        # norm isapprox, expected equal
        @test predict(_pl_fit(cfg(:pearson), P), xev) ≈ predict(_pl_fit(cfg(:pearson), P2), xev) rtol = 1e-6
        # the same transform changes a model fitted on levels
        @test cor(predict(_pl_fit(cfg(:mse), P), xev), predict(_pl_fit(cfg(:mse), P2), xev)) < 0.999

        # Any positive per-date scale and shift: rounding of the stored target may move a split,
        # so only the metric on the untransformed eval target is compared.
        a = 10 .^ (2 .* rand(rng, 60) .- 1)
        b = 40 .* rand(rng, 60) .- 20
        y3 = copy(P.y)
        y3[P.tr] .= (a[P.date] .* P.y .+ b[P.date])[P.tr]
        m1 = _pl_fit_eval(cfg(:pearson), P)
        m3 = _pl_fit_eval(cfg(:pearson), merge(P, (; y=y3)))
        @test abs(m1.info[:logger][:metrics][end] - m3.info[:logger][:metrics][end]) < F3_AFFINE_METRIC_ATOL
    end

    @testset "F4 date-constant feature" begin
        # A split that keeps whole dates together has G_l = G_r = 0 up to rounding under
        # :pearson, so on stumps any real feature wins. Under :mse the date level is what the
        # target mostly varies with, so the same feature wins.
        P = _pl_fit_panel(15)
        stump(loss) = EvoTreeRegressor(; loss, nrounds=20, max_depth=1, eta=0.1)
        share(m) = Dict(importance(m))[:dconst]
        @test share(_pl_fit(stump(:pearson), P)) == 0.0
        @test first(importance(_pl_fit(stump(:mse), P))).first == :dconst

        # Deeper nodes hold partial dates, where interactions with it are legitimate.
        deep(loss) = EvoTreeRegressor(; loss, nrounds=30, max_depth=4, eta=0.1)
        @test share(_pl_fit(deep(:pearson), P)) < F4B_MAX_SHARE
        @test first(importance(_pl_fit(deep(:mse), P))).first == :dconst
    end

    @testset "F5 learns without a plateau" begin
        P = _pl_fit_panel(16)
        cfg(loss) = EvoTreeRegressor(; loss, metric=:pearson, nrounds=300, max_depth=4, eta=0.1,
            early_stopping_rounds=10)
        mp = _pl_fit_eval(cfg(:pearson), P)
        lg = mp.info[:logger]
        # every date's prediction is the zero bias, and a flat date scores exactly zero
        @test lg[:metrics][1] == 0.0
        # A stalled step repeats the first tree, so the fit stops improving on its own training
        # dates. Held-out dates can peak at round 1 on a noisy panel, so they cannot show this.
        mi = _pl_fit(cfg(:pearson), P; x_eval=P.x[P.tr, :], y_eval=P.y[P.tr], group_eval=P.date[P.tr])
        li = mi.info[:logger]
        @test li[:best_iter] > 1
        @test li[:best_metric] > li[:metrics][2] + F5_PLATEAU_MARGIN
        mm = _pl_fit_eval(cfg(:mse), P)
        @test lg[:best_metric] > mm.info[:logger][:best_metric] + F5_MSE_MARGIN
        @test all(isfinite, predict(mp, P.x[P.ev, :]))
    end

    @testset "F6 errors" begin
        P = _pl_fit_panel(17)
        cfg = EvoTreeRegressor(; loss=:pearson, nrounds=3, max_depth=3)
        xtr, ytr, dtr = P.x[P.tr, :], P.y[P.tr], P.date[P.tr]

        e = _pl_err(() -> fit(cfg; x_train=xtr, y_train=ytr, verbosity=0))
        @test e isa ErrorException
        @test occursin("`loss = :pearson` requires group information", _pl_msg(e))

        dtrain = (date=dtr, x1=xtr[:, 1], x2=xtr[:, 2], y=ytr)
        e = _pl_err(() -> fit(cfg, dtrain; target_name=:y, verbosity=0))
        @test e isa ErrorException
        @test occursin("`loss = :pearson` requires group information", _pl_msg(e))

        e = _pl_err(() -> fit(cfg; x_train=xtr, y_train=hcat(ytr, ytr), group_train=dtr, verbosity=0))
        @test e isa ErrorException
        @test occursin("`loss = :pearson` takes a single target vector", _pl_msg(e))

        # every date constant
        e = _pl_err(() -> fit(cfg; x_train=xtr, y_train=Float64.(dtr), group_train=dtr, verbosity=0))
        @test e isa ErrorException
        @test occursin("nothing to correlate", _pl_msg(e))

        # the metric needs the eval groups
        e = _pl_err(() -> fit(cfg; x_train=xtr, y_train=ytr, group_train=dtr,
            x_eval=P.x[P.ev, :], y_eval=P.y[P.ev], verbosity=0))
        @test e isa ErrorException
        @test occursin("`metric = :pearson` requires group information", _pl_msg(e))

        @test_throws ErrorException EvoTreeMLE(; loss=:pearson)
    end

    @testset "F7 table API" begin
        P = _pl_fit_panel(18)
        cfg = EvoTreeRegressor(; loss=:pearson, nrounds=20, max_depth=4, eta=0.1)
        xtr, xev = P.x[P.tr, 1:4], P.x[P.ev, 1:4]
        mm = fit(cfg; x_train=xtr, y_train=P.y[P.tr], group_train=P.date[P.tr],
            feature_names=_PL_FEATS[1:4], verbosity=0)
        dtrain = (date=P.date[P.tr], x1=xtr[:, 1], x2=xtr[:, 2], x3=xtr[:, 3], x4=xtr[:, 4], y=P.y[P.tr])
        deval = (date=P.date[P.ev], x1=xev[:, 1], x2=xev[:, 2], x3=xev[:, 3], x4=xev[:, 4], y=P.y[P.ev])
        mt = fit(cfg, dtrain; target_name=:y, group_name=:date, deval, verbosity=0)
        @test mt.info[:feature_names] == [:x1, :x2, :x3, :x4]
        @test mt.info[:logger][:metrics][1] == 0.0
        # norm isapprox, expected equal
        @test predict(mt, deval) ≈ predict(mm, xev) rtol = F7_RTOL
    end

    @testset "F8 determinism" begin
        P = _pl_fit_panel(19)
        cfg = EvoTreeRegressor(; loss=:pearson, nrounds=30, max_depth=4, eta=0.1, rowsample=0.8,
            colsample=0.8)
        m1 = _pl_fit_eval(cfg, P)
        m2 = _pl_fit_eval(cfg, P)
        @test predict(m1, P.x[P.ev, :]) == predict(m2, P.x[P.ev, :])
        @test m1.info[:logger][:metrics] == m2.info[:logger][:metrics]
    end

    @testset "F9 continuation from an offset" begin
        # The gradient reads only the current prediction, so 50 rounds then 50 more from the
        # first model's raw predictions must reproduce a 100-round fit up to the order in which
        # tree outputs are summed.
        P = _pl_fit_panel(20)
        cfg(n) = EvoTreeRegressor(; loss=:pearson, nrounds=n, max_depth=4, eta=0.1)
        xtr, xev = P.x[P.tr, :], P.x[P.ev, :]
        m = _pl_fit(cfg(100), P)
        m1 = _pl_fit(cfg(50), P)
        m2 = _pl_fit(cfg(50), P; offset_train=predict(m1, xtr))
        # norm isapprox
        @test predict(m, xtr) ≈ predict(m1, xtr) .+ predict(m2, xtr) rtol = F9_RTOL
        @test predict(m, xev) ≈ predict(m1, xev) .+ predict(m2, xev) rtol = F9_RTOL
    end

end
