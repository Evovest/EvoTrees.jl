using Test
using Statistics
using Random
using EvoTrees
using EvoTrees: fit, predict, build_group_index, ngroups, group_rows, subsample, pearson, _pearson_group

@testset "ranking groups" begin

    @testset "group index" begin
        # Groups are supplied as a per-row id column, so the rows of a group need not be
        # contiguous, sorted, or numeric. The index must recover them regardless.
        raw = [3, 1, 3, 2, 1, 3]
        gi = build_group_index(raw)

        @test ngroups(gi) == 3
        @test length(gi) == 6
        # Ids are normalised to 1:ngroups by sorted level, so 1 -> 1, 2 -> 2, 3 -> 3.
        @test gi.group == UInt32[3, 1, 3, 2, 1, 3]
        # Every row belongs to exactly one group, and none is lost.
        @test sort(Int.(gi.rows)) == collect(1:6)
        @test sort(Int.(group_rows(gi, 1))) == [2, 5]
        @test sort(Int.(group_rows(gi, 2))) == [4]
        @test sort(Int.(group_rows(gi, 3))) == [1, 3, 6]

        # Non-numeric ids work the same way.
        gs = build_group_index(["b", "a", "b"])
        @test ngroups(gs) == 2
        @test sort(Int.(group_rows(gs, 1))) == [2]      # "a"
        @test sort(Int.(group_rows(gs, 2))) == [1, 3]   # "b"

        @test_throws ErrorException build_group_index(Int[])
    end

    @testset "ndcg" begin
        ndcg_group = EvoTrees._ndcg_group
        rel = [3.0, 2, 3, 0, 1, 2]
        desc = Float64.(collect(6:-1:1))   # scores that reproduce `rel`'s given order

        # Hand computation: gains 2^r - 1, discounts log2(i+1).
        gains = [7.0, 3, 7, 0, 1, 3]
        disc = [log2(i + 1) for i in 1:6]
        expected = sum(gains ./ disc) / sum(sort(gains; rev=true) ./ disc)
        @test ndcg_group(desc, rel, 100) ≈ expected

        @test ndcg_group(Float64.(rel), rel, 100) ≈ 1.0          # perfect ordering
        @test ndcg_group(-Float64.(rel), rel, 100) < 1.0         # reversed is worse
        @test ndcg_group([1.0], [2.0], 100) ≈ 1.0                # single document
        # A group whose documents are all irrelevant cannot be ranked wrongly, and is
        # scored 1.0 to match the convention used in the LTRC tutorial.
        @test ndcg_group([3.0, 2, 1], [0.0, 0, 0], 100) == 1.0
        # Ties in the prediction must not error.
        @test ndcg_group([1.0, 1.0, 1.0], [2.0, 1.0, 0.0], 100) isa Float64

        # Truncation is bounded by the group size, so k beyond it changes nothing.
        @test ndcg_group(desc, rel, 6) ≈ ndcg_group(desc, rel, 100)
        @test ndcg_group(desc, rel, 1) ≈ 1.0    # top document is maximally relevant
    end

    @testset "group-aware sampling" begin
        # A group is the unit a ranking objective or metric is defined over, so sampling
        # must take whole groups. A split group changes the comparison set.
        rng = Xoshiro(4)
        qid = repeat(1:60, inner=8)[randperm(rng, 480)]
        gi = build_group_index(qid)
        n = length(qid)

        for rowsample in (0.3, 0.5, 0.8)
            is, left, mask = zeros(UInt32, n), zeros(UInt32, n), zeros(UInt8, n)
            sel = subsample(left, is, mask, rowsample, Xoshiro(1), gi)
            picked = unique(gi.group[sel])
            @test !isempty(picked)
            # every picked group contributes all of its rows, and nothing else appears
            for g in picked
                @test count(==(g), gi.group[sel]) == length(group_rows(gi, g))
            end
            @test length(sel) == sum(length(group_rows(gi, g)) for g in picked)
            @test allunique(sel)
        end

        # rowsample = 1 keeps every group.
        is, left, mask = zeros(UInt32, n), zeros(UInt32, n), zeros(UInt8, n)
        sel = subsample(left, is, mask, 1.0, Xoshiro(1), gi)
        @test length(sel) == n
    end

    @testset "loss reaches the group index" begin
        # Every loss defined today is per observation and ignores groups, but a group
        # defined objective such as LambdaRank needs the index at the call site. The
        # forwarding method makes that reachable without touching any existing loss.
        struct _GroupProbeLoss <: EvoTrees.LossType end
        seen = Ref{Any}(:unset)
        # `params` is constrained the same way every loss in `src/loss.jl` constrains it;
        # leaving it untyped would be ambiguous against the forwarding method.
        EvoTrees.update_grads!(∇, p, y, ::Type{_GroupProbeLoss}, params::EvoTrees.EvoTypes, group) =
            (seen[] = group; nothing)

        gi = build_group_index([1, 1, 2, 2, 2])
        ∇ = zeros(Float32, 3, 5)
        p = zeros(Float32, 1, 5)
        EvoTrees.update_grads!(∇, p, zeros(Float32, 5), _GroupProbeLoss,
            EvoTreeRegressor(; nrounds=1), gi)
        @test seen[] === gi

        # And a loss that does not define the six argument form still works, receiving
        # nothing and dispatching to its existing method.
        ∇2 = zeros(Float32, 3, 5)
        ∇2[3, :] .= 1        # weight row; gradients are scaled by it
        p2 = zeros(Float32, 1, 5)
        EvoTrees.update_grads!(∇2, p2, Float32[1, 2, 3, 4, 5], EvoTrees.MSE,
            EvoTreeRegressor(; nrounds=1), nothing)
        @test any(!iszero, ∇2)
    end

    @testset "lambdarank gradients" begin
        # |dNDCG| is held constant wrt the scores, so the gradient must equal dC/ds for
        # C = sum over pairs rel_a > rel_b of |dNDCG_ab| * log(1 + exp(-(s_a - s_b))).
        function pair_deltas(scores, rel, k)
            n = length(rel)
            ideal = sort(rel; rev=true)
            maxdcg = sum((2.0^ideal[i] - 1) / log2(i + 1) for i in 1:min(k, n))
            ord = sortperm(scores; rev=true)
            rank = zeros(Int, n)
            for pos in 1:n
                rank[ord[pos]] = pos
            end
            disc = [pos <= k ? 1 / log2(pos + 1) : 0.0 for pos in 1:n]
            D = zeros(n, n)
            for a in 1:n, b in 1:n
                rel[a] > rel[b] || continue
                D[a, b] = abs((2.0^rel[a] - 2.0^rel[b]) *
                              (disc[rank[a]] - disc[rank[b]])) / maxdcg
            end
            D
        end
        pair_cost(s, D, rel) = sum(
            D[a, b] * log(1 + exp(-(s[a] - s[b])))
            for a in eachindex(rel), b in eachindex(rel) if rel[a] > rel[b])

        rng = Xoshiro(3)
        for _ in 1:3, k in (3, 10)
            n = 7
            rel = Float64.(rand(rng, 0:4, n))
            length(unique(rel)) < 2 && continue
            sc = randn(rng, n)
            D = pair_deltas(sc, rel, k)
            ∇ = zeros(Float32, 3, n)
            ∇[3, :] .= 1
            EvoTrees._lambdarank_group!(EvoTrees.LambdaRankScratch(), ∇,
                reshape(Float32.(sc), 1, n), Float32.(rel), collect(1:n), k)
            h = 1e-6
            for i in 1:n
                sp = copy(sc); sp[i] += h
                sm = copy(sc); sm[i] -= h
                @test ∇[1, i] ≈ (pair_cost(sp, D, rel) - pair_cost(sm, D, rel)) / (2h) atol = 1e-5
            end
            @test all(∇[2, :] .>= 0)
            @test all(isfinite, ∇[1:2, :])
            # Pairwise contributions are antisymmetric, so a query's gradients cancel.
            @test sum(∇[1, :]) ≈ 0 atol = 1e-5
        end

        # A query with no relevant document carries no ranking signal.
        ∇0 = zeros(Float32, 3, 4)
        ∇0[3, :] .= 1
        EvoTrees._lambdarank_group!(EvoTrees.LambdaRankScratch(), ∇0,
            reshape(Float32[4, 3, 2, 1], 1, 4), Float32[0, 0, 0, 0], collect(1:4), 10)
        @test all(iszero, ∇0[1:2, :])
    end

    @testset "xendcg gradients" begin
        function xe_loss(s, grade, off)
            local t = (exp2.(grade) .- off) ./ sum(exp2.(grade) .- off)
            local smax = maximum(s)
            return -sum(t .* (s .- smax .- log(sum(exp.(s .- smax)))))
        end
        function xe_grads(s, grade, off; q=ones(Int, length(s)), w=ones(length(s)))
            local out = zeros(Float32, 3, length(s))
            out[3, :] .= w
            EvoTrees._xendcg_grads!(out, reshape(Float32.(s), 1, :), Float32.(grade), build_group_index(q), off)
            return out
        end

        rng = Xoshiro(5)
        for _ in 1:3, off in (0.0, 0.5, 1.0)
            n = 9
            grade = Float64.(rand(rng, 0:4, n))
            s = randn(rng, n)
            ∇ = xe_grads(s, grade, off)
            h = 1e-6
            for i in 1:n
                e = h .* (1:n .== i)
                @test ∇[1, i] ≈ (xe_loss(s .+ e, grade, off) - xe_loss(s .- e, grade, off)) / (2h) atol = 1e-6
            end
            ρ = exp.(s) ./ sum(exp.(s))
            @test all(isapprox.(∇[2, :], ρ .* (1 .- ρ); atol=1e-6))
            @test sum(∇[1, :]) ≈ 0 atol = 1e-6
            @test all(isapprox.(xe_grads(s .+ 3, grade, off), ∇; atol=1e-6))
            perm = randperm(rng, n)
            @test all(isapprox.(xe_grads(s[perm], grade[perm], off), ∇[:, perm]; atol=1e-6))
        end

        # At equal scores the gradient is 1/n minus the target.
        @test xe_grads(zeros(4), [0.0, 0, 1, 2], 1.0)[1, :] ≈ [0.25, 0.25, 0, -0.5]
        @test xe_grads(zeros(4), [0.0, 0, 1, 2], 0.0)[1, :] ≈ 0.25 .- [1, 1, 2, 4] ./ 8

        q = [1, 2, 1, 2, 2, 1, 2]
        s = randn(rng, 7)
        gr = Float64[3, 0, 1, 2, 2, 0, 4]
        ∇ = xe_grads(s, gr, 0.5; q)
        for g in 1:2
            @test sum(∇[1, q.==g]) ≈ 0 atol = 1e-6
            @test all(isapprox.(∇[:, q.==g], xe_grads(s[q.==g], gr[q.==g], 0.5); atol=1e-6))
        end

        # A query weighs by the mean of its rows' weights: 8/3 for query 1, 2 for query 2.
        wv = [1.0, 2, 3, 1, 1, 4, 4]
        ∇w = xe_grads(s, gr, 0.5; q, w=wv)
        @test all(isapprox.(∇w[1:2, q.==1], ∇[1:2, q.==1] .* (8 / 3); atol=1e-6))
        @test all(isapprox.(∇w[1:2, q.==2], ∇[1:2, q.==2] .* 2; atol=1e-6))

        big = xe_grads([1000.0, 999, 0], [2.0, 1, 0], 0.5)
        @test all(isfinite, big)
        @test all(isapprox.(big, xe_grads([0.0, -1, -1000], [2.0, 1, 0], 0.5); atol=1e-6))

        ∇big = xe_grads([0.0, 0, 0], [2000.0, 1999, 0], 0.5)
        @test all(isfinite, ∇big)
        @test ∇big[1, :] ≈ [1 / 3 - 2 / 3, 1 / 3 - 1 / 3, 1 / 3] atol = 1e-6

        @test all(iszero, xe_grads([1.0], [2.0], 0.5)[1:2, :])
        @test all(iszero, xe_grads([1.0, 2.0, 3.0], [0.0, 0, 0], 1.0)[1:2, :])
    end

    @testset "xendcg metric" begin
        q = [1, 1, 1, 2, 2, 2, 2, 3]
        gr = Float32[2, 0, 1, 3, 3, 0, 1, 2]
        pv = Float32[0.5, -1, 0.2, 1.5, 0.3, -0.4, 0.0, 9.0]
        wv = Float32[1, 1, 1, 3, 3, 3, 3, 5]
        gi = build_group_index(q)
        function ce(s, grade, off)
            local t = (exp2.(grade) .- off) ./ sum(exp2.(grade) .- off)
            return -sum(t .* log.(exp.(s) ./ sum(exp.(s))))
        end
        for off in (0.0, 0.5, 1.0)
            got = EvoTrees.xendcg(reshape(pv, 1, :), gr, wv, Float32[]; group=gi, xendcg_offset=off)
            # the single-row group 3 is left out
            c1 = ce(Float64.(pv[1:3]), Float64.(gr[1:3]), off)
            c2 = ce(Float64.(pv[4:7]), Float64.(gr[4:7]), off)
            @test got ≈ (c1 * 1 + c2 * 3) / 4 rtol = 1e-6
        end
        q0 = [1, 1, 1, 2, 2]
        p0 = Float32[0.5, -1, 0.2, 3, 1]
        g0 = Float32[2, 0, 1, 0, 0]
        @test EvoTrees.xendcg(reshape(p0, 1, :), g0, ones(Float32, 5), Float32[];
            group=build_group_index(q0), xendcg_offset=1.0) ≈ ce(Float64.(p0[1:3]), Float64.(g0[1:3]), 1.0) rtol = 1e-6
        @test EvoTrees.xendcg(Float32[1000 999 0], Float32[2, 1, 0], ones(Float32, 3), Float32[];
            group=build_group_index(ones(Int, 3))) ≈ EvoTrees.xendcg(Float32[0 -1 -1000], Float32[2, 1, 0],
            ones(Float32, 3), Float32[]; group=build_group_index(ones(Int, 3))) rtol = 1e-6
        t = (exp2.(Float64.(gr[4:7])) .- 0.5) ./ sum(exp2.(Float64.(gr[4:7])) .- 0.5)
        best = EvoTrees.xendcg(reshape(Float32.(log.(t)), 1, :), gr[4:7], ones(Float32, 4), Float32[];
            group=build_group_index(ones(Int, 4)))
        @test best ≈ -sum(t .* log.(t)) rtol = 1e-5
        @test best < EvoTrees.xendcg(reshape(Float32.(log.(t)) .+ Float32[0.1, 0, 0, 0], 1, :), gr[4:7],
            ones(Float32, 4), Float32[]; group=build_group_index(ones(Int, 4)))
        @test !EvoTrees.is_maximise(EvoTrees.xendcg)
        @test_throws ErrorException EvoTrees.xendcg(reshape(pv, 1, :), gr, wv, Float32[])
        @test_throws ErrorException EvoTrees.xendcg(reshape(pv, 1, :), gr .- 1, wv, Float32[]; group=gi)
        @test_throws ErrorException EvoTrees.xendcg(reshape(pv, 1, :), gr, wv, Float32[];
            group=build_group_index(collect(1:8)))
    end

    @testset "eval-only grouping" begin
        # Training with the usual per-row sampling while evaluating a group-aware metric.
        # `eval_group_name` defaults to `group_name`, and set on its own it leaves training
        # ungrouped.
        rng = Xoshiro(21)
        nobs = 1_500
        q = repeat(1:150, inner=10)
        x = rand(rng, nobs, 3)
        y = clamp.(round.(2 .* x[:, 1] .+ randn(rng, nobs)), 0, 4)
        tr, te = 1:1_000, 1_001:nobs
        dtrain = (q=q[tr], f1=x[tr, 1], f2=x[tr, 2], f3=x[tr, 3], y=y[tr])
        deval = (q=q[te], f1=x[te, 1], f2=x[te, 2], f3=x[te, 3], y=y[te])

        cfg = EvoTreeRegressor(; loss=:mse, metric=:ndcg, ndcg_k=10, nrounds=20,
            max_depth=4, rowsample=0.5)
        m = fit(cfg, dtrain; target_name=:y, eval_group_name=:q, deval, verbosity=0)

        # training saw no groups, so `q` is just another feature unless excluded by name
        @test :q in m.info[:feature_names]
        mets = m.info[:logger][:metrics]
        @test all(0 .<= mets .<= 1)
        @test length(mets) == 21

        # and it still defaults to `group_name` when only that is given
        m2 = fit(cfg, dtrain; target_name=:y, group_name=:q, deval, verbosity=0)
        @test m2.info[:group_name] == :q
        @test all(0 .<= m2.info[:logger][:metrics] .<= 1)
    end

    @testset "fit with lambdarank" begin
        # Each query grades on its own curve, so absolute label level is query-specific
        # noise that regression must fit but a ranking objective can ignore.
        rng = Xoshiro(7)
        qid = Int[]
        rows = Vector{Float64}[]
        rel = Float64[]
        for q in 1:600
            shift = 2.5 * randn(rng)
            for _ in 1:rand(rng, 8:16)
                f = randn(rng, 4)
                sc = 1.5f[1] - f[2] + 0.5f[1] * f[3] + randn(rng)
                push!(qid, q)
                push!(rows, f)
                push!(rel, clamp(round(sc + shift + 2), 0, 4))
            end
        end
        x = permutedims(reduce(hcat, rows))
        y = rel
        tr, te = qid .<= 450, qid .> 450

        # Both report `:ndcg` so the two are comparable; `:mse` would otherwise default to
        # reporting squared error.
        cfg(loss) = EvoTreeRegressor(; loss, metric=:ndcg, ndcg_k=10, nrounds=150,
            max_depth=5, eta=0.05, rowsample=0.7)
        fitted(loss) = fit(cfg(loss); x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid[te], verbosity=0)

        m = fitted(:lambdarank)
        @test EvoTreeRegressor(; loss=:lambdarank).metric == :ndcg
        mets = m.info[:logger][:metrics]
        @test all(0 .<= mets .<= 1)
        @test mets[end] > mets[1]

        # Whether the ranking objective beats regression is an empirical property that
        # depends on the data and sample size, so it belongs in a benchmark rather than
        # here. What is asserted is that it optimises something different: the two produce
        # materially different models on the same inputs.
        pl = predict(m, x[te, :])
        pm = predict(fitted(:mse), x[te, :])
        @test cor(pl, pm) < 0.999

        @test_throws ErrorException fit(
            EvoTreeRegressor(; loss=:lambdarank, nrounds=5, max_depth=3);
            x_train=x[tr, :], y_train=y[tr], verbosity=0)
        @test_throws AssertionError fit(
            EvoTreeRegressor(; loss=:lambdarank, nrounds=5, max_depth=3);
            x_train=x[tr, :], y_train=y[tr] .- 1, group_train=qid[tr], verbosity=0)

        cfg_xe(; kw...) = EvoTreeRegressor(; loss=:xendcg, nrounds=150, max_depth=5, eta=0.05,
            rowsample=0.7, kw...)
        fit_xe(cfg) = fit(cfg; x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid[te], verbosity=0)
        @test EvoTreeRegressor(; loss=:xendcg).metric == :xendcg
        @test EvoTreeRegressor(; loss=:xendcg).xendcg_offset == 0.5
        mx = fit_xe(cfg_xe())
        mets = mx.info[:logger][:metrics]
        @test mets[end] < mets[1]
        mn = fit_xe(cfg_xe(; metric=:ndcg, ndcg_k=10))
        @test mn.info[:logger][:metrics][end] > mn.info[:logger][:metrics][1]
        px = predict(mx, x[te, :])
        @test cor(px, pm) < 0.999
        @test cor(px, pl) < 0.999
        m1 = fit_xe(cfg_xe(; xendcg_offset=1.0))
        @test predict(m1, x[te, :]) != px
        @test predict(fit_xe(cfg_xe()), x[te, :]) == px
        logged_vs_direct(m, xe) = (m.info[:logger][:metrics][end],
            EvoTrees.xendcg(reshape(Float32.(predict(m, xe)), 1, :), Float32.(y[te]), ones(Float32, sum(te)),
                Float32[]; group=build_group_index(qid[te]), xendcg_offset=1.0))
        @test ≈(logged_vs_direct(m1, x[te, :])...; rtol=1e-4)
        @test predict(fit(cfg_xe(); x_train=x[tr, :], y_train=reshape(y[tr], :, 1), group_train=qid[tr],
            x_eval=x[te, :], y_eval=reshape(y[te], :, 1), group_eval=qid[te], verbosity=0), x[te, :]) == px

        @test_throws ErrorException EvoTreeRegressor(; loss=:xendcg, xendcg_offset=1.5)
        @test_throws ErrorException EvoTreeRegressor(; loss=:xendcg, xendcg_offset=-0.1)
        @test_throws ErrorException fit(cfg_xe(; nrounds=5); x_train=x[tr, :], y_train=y[tr], verbosity=0)
        @test_throws AssertionError fit(cfg_xe(; nrounds=5);
            x_train=x[tr, :], y_train=y[tr] .- 1, group_train=qid[tr], verbosity=0)
        for loss in (:xendcg, :lambdarank)
            @test_throws ErrorException fit(EvoTreeRegressor(; loss, nrounds=5);
                x_train=x[tr, :], y_train=hcat(y[tr], y[tr]), group_train=qid[tr], verbosity=0)
        end

        dtr = (q=qid[tr], f1=x[tr, 1], f2=x[tr, 2], f3=x[tr, 3], f4=x[tr, 4], y=y[tr])
        dte = (q=qid[te], f1=x[te, 1], f2=x[te, 2], f3=x[te, 3], f4=x[te, 4], y=y[te])
        mt = fit(cfg_xe(), dtr; target_name=:y, group_name=:q, deval=dte, verbosity=0)
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test mt.info[:logger][:metrics][end] < mt.info[:logger][:metrics][1]
        mt1 = fit(cfg_xe(; xendcg_offset=1.0), dtr; target_name=:y, group_name=:q, deval=dte, verbosity=0)
        @test ≈(logged_vs_direct(mt1, dte)...; rtol=1e-4)
    end

    @testset "fit with groups" begin
        rng = Xoshiro(42)
        nq = 300
        qid = Int[]
        rows = Vector{Float64}[]
        rel = Float64[]
        for q in 1:nq
            for _ in 1:rand(rng, 8:16)
                f = randn(rng, 4)
                s = 1.5f[1] - f[2] + 0.5f[1] * f[3] + 1.2randn(rng)
                push!(qid, q)
                push!(rows, f)
                push!(rel, clamp(round(s + 2), 0, 4))
            end
        end
        x = permutedims(reduce(hcat, rows))
        y = rel
        tr = qid .<= 220
        te = .!tr

        # Matrix API.
        m = fit(
            EvoTreeRegressor(; nrounds=80, max_depth=5, eta=0.05, rowsample=0.7,
                metric=:ndcg, ndcg_k=10);
            x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid[te], verbosity=0)
        mets = m.info[:logger][:metrics]
        @test all(0 .<= mets .<= 1)
        @test mets[end] > mets[1]

        # Table API. The group column must not be picked up as a feature.
        dtrain = (q=qid[tr], f1=x[tr, 1], f2=x[tr, 2], f3=x[tr, 3], f4=x[tr, 4], y=y[tr])
        deval = (q=qid[te], f1=x[te, 1], f2=x[te, 2], f3=x[te, 3], f4=x[te, 4], y=y[te])
        mt = fit(
            EvoTreeRegressor(; nrounds=80, max_depth=5, eta=0.05, rowsample=0.7,
                metric=:ndcg, ndcg_k=10),
            dtrain; target_name=:y, group_name=:q, deval, verbosity=0)
        @test mt.info[:feature_names] == [:f1, :f2, :f3, :f4]
        @test mt.info[:group_name] == :q
        @test mt.info[:logger][:metrics][end] > mt.info[:logger][:metrics][1]

        # A group column of the wrong length must be rejected. A short one would otherwise
        # silently train or score on a subset and report the result as if it covered
        # everything; a long one indexes past the end of the predictions.
        ntr, nte = sum(tr), sum(te)
        @test_throws ErrorException fit(
            EvoTreeRegressor(; nrounds=5, max_depth=3, metric=:ndcg);
            x_train=x[tr, :], y_train=y[tr], group_train=qid[tr][1:(ntr÷2)],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid[te], verbosity=0)
        @test_throws ErrorException fit(
            EvoTreeRegressor(; nrounds=5, max_depth=3, metric=:ndcg);
            x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid[te][1:(nte÷2)], verbosity=0)
        @test_throws ErrorException fit(
            EvoTreeRegressor(; nrounds=5, max_depth=3, metric=:ndcg);
            x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
            x_eval=x[te, :], y_eval=y[te], group_eval=qid, verbosity=0)

        # Without groups there is nothing to rank within, so `:ndcg` must say so rather
        # than silently scoring the whole eval set as one list.
        @test_throws ErrorException fit(
            EvoTreeRegressor(; nrounds=5, max_depth=3, metric=:ndcg),
            dtrain; target_name=:y, deval, verbosity=0)

        # The reported metric must equal the per-group NDCG a user would compute by hand,
        # which is what the LTRC tutorial does with a `groupby`. This is the check that the
        # metric is a ranking metric rather than a global one.
        # A group's weight is the mean of its rows' weights. The two groups must score
        # differently and the weights must not be uniform, otherwise every candidate rule
        # gives the same answer and the assertion pins nothing.
        let
            q = [1, 1, 1, 2, 2]
            rel = Float32[3, 1, 0, 2, 0]
            pr = reshape(Float32[1, 2, 3, 2, 1], 1, 5)   # group 1 ranked badly, group 2 perfectly
            gi = build_group_index(q)
            wv = Float32[1, 1, 4, 3, 3]
            got = EvoTrees.ndcg(pr, rel, wv, Float32[]; group=gi, ndcg_k=10)
            s1 = EvoTrees._ndcg_group(Float64[1, 2, 3], Float64[3, 1, 0], 10)
            s2 = EvoTrees._ndcg_group(Float64[2, 1], Float64[2, 0], 10)
            @test s1 != s2
            agg(f) = (s1 * f(Float64[1, 1, 4]) + s2 * f(Float64[3, 3])) / (f(Float64[1, 1, 4]) + f(Float64[3, 3]))
            @test got ≈ agg(mean) rtol = 1e-5
            # and the rule is pinned: sum, maximum and minimum all give something else
            for other in (sum, maximum, minimum)
                @test !isapprox(got, agg(other); rtol = 1e-3)
            end
        end

        function tutorial_ndcg(p, target, k=10)
            k = min(k, length(p))
            p_order = partialsortperm(p, 1:k; rev=true)
            gains = 2 .^ target[p_order] .- 1
            discounts = log2.((1:k) .+ 1)
            dcg = sum(gains ./ discounts)
            y_order = partialsortperm(target, 1:k; rev=true)
            idcg = sum((2 .^ target[y_order] .- 1) ./ discounts)
            return idcg == 0 ? 1.0 : dcg / idcg
        end

        for k in (5, 10, typemax(Int))
            mk = fit(
                EvoTreeRegressor(; nrounds=30, max_depth=4, eta=0.1, metric=:ndcg, ndcg_k=k);
                x_train=x[tr, :], y_train=y[tr], group_train=qid[tr],
                x_eval=x[te, :], y_eval=y[te], group_eval=qid[te], verbosity=0)
            p = predict(mk, x[te, :])
            ye, qe = y[te], qid[te]
            manual = mean(tutorial_ndcg(p[qe.==q], ye[qe.==q], k) for q in unique(qe))
            @test mk.info[:logger][:metrics][end] ≈ manual rtol = 1e-6
        end
    end


    @testset "grouped pearson" begin
        # Weighted Pearson within each group, averaged over groups. Unlike `:ndcg` a row's own
        # weight enters its group's score, so both levels of weighting are live.
        function wpearson(x, y, w)
            sw = sum(w)
            mx, my = sum(w .* x) / sw, sum(w .* y) / sw
            cxy = sum(w .* (x .- mx) .* (y .- my)) / sw
            vx = sum(w .* (x .- mx) .^ 2) / sw
            vy = sum(w .* (y .- my) .^ 2) / sw
            return cxy / sqrt(vx * vy)
        end

        q = UInt32[1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        pv = Float64[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
        yv = Float64[2, 1, 5, 1, 6, 1, 9, 4, 5, 4, 4, 9]
        wv = Float64[1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 4, 2]
        gi = build_group_index(q)
        for w in (ones(length(q)), wv)
            got = pearson(reshape(Float32.(pv), 1, :), Float32.(yv), Float32.(w), Float32[]; group=gi)
            cs = [wpearson(pv[q.==g], yv[q.==g], w[q.==g]) for g in 1:3]
            ws = [mean(w[q.==g]) for g in 1:3]
            @test got ≈ sum(cs .* ws) / sum(ws) rtol = 1e-5
        end
        # per-row weights move the score, which is the difference from `:ndcg`
        unit = pearson(reshape(Float32.(pv), 1, :), Float32.(yv), ones(Float32, 12), Float32[]; group=gi)
        wtd = pearson(reshape(Float32.(pv), 1, :), Float32.(yv), Float32.(wv), Float32[]; group=gi)
        @test unit != wtd

        # groups carrying no signal are left out rather than averaged in as zero
        @test isnothing(_pearson_group(reshape([1.0], 1, 1), [2.0], [1.0], 1:1, 1))
        @test isnothing(_pearson_group(reshape(Float64[1, 2, 3], 1, 3), Float64[5, 5, 5], ones(3), 1:3, 1))
        @test _pearson_group(reshape(Float64[2, 2, 2], 1, 3), Float64[1, 2, 3], ones(3), 1:3, 1) == 0.0
        q2 = UInt32[1, 1, 1, 2, 2, 2]
        y2 = Float32[1, 2, 3, 7, 7, 7]
        got2 = pearson(reshape(Float32[1, 2, 3, 1, 2, 3], 1, :), y2, ones(Float32, 6), Float32[];
            group=build_group_index(q2))
        @test got2 ≈ 1.0 rtol = 1e-6

        # A two-parameter likelihood interleaves location and scale per target, so target 2
        # predicts into row 3, not row 2. Row 2 here is anti-correlated with target 2 inside
        # every group, so reading it would score 0.0 rather than 1.0.
        qm = UInt32[1, 1, 1, 2, 2, 2]
        ym = Float32[1 2 3 4 5 6; 6 5 4 3 2 1]
        pm = Float32[1 2 3 4 5 6; 1 2 3 1 2 3; 6 5 4 3 2 1; 0 0 0 0 0 0]
        @test pearson(pm, ym, ones(Float32, 6), Float32[]; group=build_group_index(qm)) ≈ 1.0 rtol = 1e-6
        # a single-target MLE model is the K == 1 case of the same rule
        @test pearson(Float32[1 2 3 4 5 6; 9 9 9 9 9 9], Float32[1, 2, 3, 4, 5, 6],
            ones(Float32, 6), Float32[]; group=build_group_index(qm)) ≈ 1.0 rtol = 1e-6
        @test EvoTrees.is_maximise(pearson)
        # without groups there is nothing to correlate within
        @test_throws ErrorException pearson(reshape(Float32.(pv), 1, :), Float32.(yv),
            ones(Float32, 12), Float32[])

        # the maintainer's case: train ungrouped, track a grouped Pearson
        rng = Xoshiro(2)
        nobs = 1_500
        g = repeat(1:150, inner=10)
        x = rand(rng, nobs, 3)
        y = 2 .* x[:, 1] .+ 0.3 .* randn(rng, nobs)
        tr, te = 1:1_000, 1_001:nobs
        dtr = (q=g[tr], f1=x[tr, 1], f2=x[tr, 2], f3=x[tr, 3], y=y[tr])
        dev = (q=g[te], f1=x[te, 1], f2=x[te, 2], f3=x[te, 3], y=y[te])
        for cfg in (EvoTreeRegressor(loss=:mse, metric=:pearson, nrounds=25, max_depth=4, rowsample=0.5),
            EvoTreeMLE(loss=:gaussian_mle, metric=:pearson, nrounds=20, max_depth=4))
            m = fit(cfg, dtr; target_name=:y, eval_group_name=:q, deval=dev, verbosity=0)
            mets = m.info[:logger][:metrics]
            @test all(-1 .<= mets .<= 1)
            @test mets[end] > 0.5
        end

        # the same, with two targets: the second must be scored against its own location
        y2 = -3 .* x[:, 2] .+ 0.3 .* randn(rng, nobs)
        dtr2 = (q=g[tr], f1=x[tr, 1], f2=x[tr, 2], f3=x[tr, 3], y=y[tr], y2=y2[tr])
        dev2 = (q=g[te], f1=x[te, 1], f2=x[te, 2], f3=x[te, 3], y=y[te], y2=y2[te])
        mmt = fit(EvoTreeMLE(loss=:gaussian_mle, metric=:pearson, nrounds=20, max_depth=4), dtr2;
            target_name=["y", "y2"], eval_group_name=:q, deval=dev2, verbosity=0)
        @test size(predict(mmt, dev2), 2) == 4
        @test mmt.info[:logger][:metrics][end] > 0.8
    end

end
