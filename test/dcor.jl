using EvoTrees: dcov2, dcor2, dcov2_grad, DcorCache

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
