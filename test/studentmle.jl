using Test
using Statistics
using Random
using EvoTrees
using EvoTrees: fit, predict, mle2p_grad_hess, StudentMLE, GaussianMLE, LogisticMLE

# Student-t MLE loss (μ, log σ) with fixed ν.
# The testsets are ordered by which part of the patch they exercise, so this file
# doubles as a progress checklist while the change is being landed.
@testset "student_mle" begin

    ##########################################################################
    # 1. KERNEL — src/loss.jl only. Pure math, no fitting.
    ##########################################################################
    @testset "kernel: gradients match finite differences" begin
        ν = 4.0
        nll(μ, ls, y) = ls + (ν + 1) / 2 * log1p((μ - y)^2 * exp(-2ls) / ν)
        h1 = 1e-6      # first derivatives:  ~eps^(1/3)
        h2 = 1e-4      # second derivative:  ~eps^(1/4). The central second difference
                       # divides by h², so cancellation noise scales as eps/h². At
                       # μ == y the NLL reduces to `ls` exactly and h1 would leave
                       # nothing but rounding.
        for (μ, ls, y) in [(0.1, -0.3, 0.4), (-1.0, 0.2, 2.5), (0.0, 0.0, -8.0), (2.0, 1.0, 2.0)]
            g_mu, g_ls, hess_mu, hess_ls = mle2p_grad_hess(StudentMLE, μ, ls, y, ν)
            @test g_mu ≈ (nll(μ + h1, ls, y) - nll(μ - h1, ls, y)) / 2h1 rtol = 1e-4 atol = 1e-9
            @test g_ls ≈ (nll(μ, ls + h1, y) - nll(μ, ls - h1, y)) / 2h1 rtol = 1e-4 atol = 1e-9
            # hess_ls is the observed second derivative; hess_mu is deliberately the
            # Fisher information, so only hess_ls is expected to match a difference.
            @test hess_ls ≈ (nll(μ, ls + h2, y) - 2nll(μ, ls, y) + nll(μ, ls - h2, y)) / h2^2 rtol = 1e-2 atol = 1e-6
        end
    end

    @testset "kernel: hessians stay positive (get_gain denominator)" begin
        # The OBSERVED h_μ turns negative past |z| > √ν. Fisher must not.
        for ν in (2.5, 3.0, 4.0, 8.0), z in (0.0, 1.0, 3.0, 10.0, 100.0)
            _, _, hess_mu, hess_ls = mle2p_grad_hess(StudentMLE, z, 0.0, 0.0, ν)
            @test hess_mu > 0
            @test hess_ls >= 0
            @test isfinite(hess_mu) && isfinite(hess_ls)
        end
    end

    @testset "kernel: ν → ∞ collapses onto GaussianMLE" begin
        for (μ, ls, y) in [(0.1, -0.3, 0.4), (-1.0, 0.2, 2.5), (0.5, 0.5, -0.5)]
            s = mle2p_grad_hess(StudentMLE, μ, ls, y, 1e9)
            g = mle2p_grad_hess(GaussianMLE, μ, ls, y)
            @test all(isapprox.(s, g; rtol = 1e-5, atol = 1e-9))
        end
    end

    @testset "kernel: score is steeper in the bulk, redescends in the tail" begin
        ν = 4.0
        # Near zero the weight a = (ν+1)/(ν+u) → (ν+1)/ν, so the t score is STEEPER
        # than the Gaussian by 25% at ν=4. The t does not merely clip outliers, it
        # reallocates influence from the tails into the bulk.
        gs0, = mle2p_grad_hess(StudentMLE, 0.0, 0.0, -0.1, ν)
        gg0, = mle2p_grad_hess(GaussianMLE, 0.0, 0.0, -0.1)
        @test gs0 / gg0 ≈ (ν + 1) / (ν + 0.01) rtol = 1e-6

        # Far out it redescends well below the Gaussian, which grows without bound.
        gs, = mle2p_grad_hess(StudentMLE, 0.0, 0.0, -20.0, ν)
        gg, = mle2p_grad_hess(GaussianMLE, 0.0, 0.0, -20.0)
        @test abs(gs) < abs(gg) / 10

        # The score peaks at |z| = √ν with value (ν+1)/(2√ν), then declines. Both
        # follow from d/dz [z(ν+1)/(ν+z²)] = (ν+1)(ν-z²)/(ν+z²)².
        zs = 0.05:0.05:20.0
        scores = [abs(mle2p_grad_hess(StudentMLE, 0.0, 0.0, -z, ν)[1]) for z in zs]
        @test zs[argmax(scores)] ≈ sqrt(ν) atol = 0.05
        @test maximum(scores) ≈ (ν + 1) / (2 * sqrt(ν)) rtol = 1e-3
        @test issorted(scores[argmax(scores):end]; rev = true)
    end

    @testset "kernel: existing MLE losses still resolve (5-arg forwarders)" begin
        # Regression guard: update_grads! now always passes ν, so the 4-arg-only
        # methods would MethodError on every gaussian_mle / logistic_mle fit.
        @test mle2p_grad_hess(GaussianMLE, 0.1, -0.3, 0.4, 4.0) ==
              mle2p_grad_hess(GaussianMLE, 0.1, -0.3, 0.4)
        @test mle2p_grad_hess(LogisticMLE, 0.1, -0.3, 0.4, 4.0) ==
              mle2p_grad_hess(LogisticMLE, 0.1, -0.3, 0.4)
    end

    ##########################################################################
    # 2. FIT — src/init.jl. Without the StudentMLE branch K defaults to 1 and
    #    the model trains to a constant without erroring.
    ##########################################################################
    Random.seed!(123)
    nobs = 20_000
    x = randn(nobs, 3)
    signal = 2 .* x[:, 1]
    y_gauss = signal .+ 0.5 .* randn(nobs)

    @testset "fit: two output heads and a learned μ" begin
        m = fit(EvoTreeMLE(; loss=:student_mle, nu=4.0, nrounds=200, eta=0.05);
                x_train=x, y_train=y_gauss, verbosity=0)
        p = predict(m, x)
        @test m.K == 2
        @test size(p, 2) == 2
        @test cor(p[:, 1], signal) > 0.95
        @test m.info[:nu] == 4.0          # ν persisted into the fitted model
    end

    ##########################################################################
    # 3. PREDICT — src/predict.jl. Missing StudentMLE in the exp list returns
    #    log σ instead of σ, silently.
    ##########################################################################
    @testset "predict: second head is σ, not log σ" begin
        m = fit(EvoTreeMLE(; loss=:student_mle, nu=4.0, nrounds=200, eta=0.05);
                x_train=x, y_train=y_gauss, verbosity=0)
        p = predict(m, x)
        @test all(p[:, 2] .> 0)
        # MLE t(4) scale on Gaussian residuals of sd 0.5 is ≈ 0.41 — below the sd
        # because the fatter tails need less scale to cover the same spread.
        @test 0.25 < median(p[:, 2]) < 0.60
    end

    ##########################################################################
    # 4. METRIC — src/metrics.jl + src/callback.jl. If ν never reaches the
    #    metric, every ν scores identically and a ν grid is meaningless.
    ##########################################################################
    @testset "metric: eval runs and ν changes the score" begin
        xe = randn(5_000, 3)
        ye = 2 .* xe[:, 1] .+ 0.5 .* randn(5_000)
        scores = map((3.0, 4.0, 12.0)) do ν
            m = fit(EvoTreeMLE(; loss=:student_mle, nu=ν, nrounds=50, eta=0.05);
                    x_train=x, y_train=y_gauss, x_eval=xe, y_eval=ye, verbosity=0)
            m.info[:logger][:metrics][end]
        end
        @test all(isfinite, scores)
        @test length(unique(scores)) == 3        # ν is actually threaded through
    end

    ##########################################################################
    # 5. RECOVERY — the σ head should find the true scale on genuine t data.
    ##########################################################################
    @testset "recovery: σ head on t(4)-distributed residuals" begin
        Random.seed!(7)
        n = 40_000
        xt = randn(n, 3)
        # exact t(4) without a Distributions dependency: chisq(4) = Σ of 4 squared normals
        chi4 = vec(sum(randn(n, 4) .^ 2; dims=2))
        t4 = randn(n) ./ sqrt.(chi4 ./ 4)
        σ_true = 0.5
        yt = 2 .* xt[:, 1] .+ σ_true .* t4

        m = fit(EvoTreeMLE(; loss=:student_mle, nu=4.0, nrounds=300, eta=0.05);
                x_train=xt, y_train=yt, verbosity=0)
        p = predict(m, xt)
        @test cor(p[:, 1], 2 .* xt[:, 1]) > 0.95
        @test 0.8 * σ_true < median(p[:, 2]) < 1.25 * σ_true
    end

    ##########################################################################
    # 6. NO REGRESSION — the other two MLE losses must be untouched end to end.
    ##########################################################################
    @testset "no regression: gaussian_mle and logistic_mle still fit" begin
        for loss in (:gaussian_mle, :logistic_mle)
            m = fit(EvoTreeMLE(; loss, nrounds=100, eta=0.05);
                    x_train=x, y_train=y_gauss, verbosity=0)
            p = predict(m, x)
            @test size(p, 2) == 2
            @test all(p[:, 2] .> 0)
            @test cor(p[:, 1], signal) > 0.95
        end
    end

end
