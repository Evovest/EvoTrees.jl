using Statistics
using StatsBase: sample
using EvoTrees: sigmoid, logit
using EvoTrees: check_args, check_parameter
using Random: seed!

# prepare a dataset
seed!(123)
features = rand(1_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

@testset "EvoTreeRegressor - MSE" begin
    # mse
    params1 = EvoTreeRegressor(
        loss=:mse,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - MSE early stopping tolerance" begin
    seed!(321)
    n = 1_000
    x = randn(n, 1)
    y = x[:, 1] .+ 0.1 .* randn(n)
    idx = sample(1:n, n, replace=false)
    n_train = floor(Int, 0.8 * n)
    x_train_es, y_train_es = x[idx[1:n_train], :], y[idx[1:n_train]]
    x_eval_es, y_eval_es = x[idx[n_train+1:end], :], y[idx[n_train+1:end]]

    params_strict = EvoTreeRegressor(
        loss=:mse,
        nrounds=50,
        early_stopping_rounds=5,
        early_stopping_tolerance=0.0,
        eta=0.05,
        seed=123,
    )
    params_tolerant = EvoTreeRegressor(
        loss=:mse,
        nrounds=50,
        early_stopping_rounds=5,
        early_stopping_tolerance=0.01,
        eta=0.05,
        seed=123,
    )

    m_strict = fit(params_strict; x_train=x_train_es, y_train=y_train_es, x_eval=x_eval_es, y_eval=y_eval_es, verbosity=0)
    m_tolerant = fit(params_tolerant; x_train=x_train_es, y_train=y_train_es, x_eval=x_eval_es, y_eval=y_eval_es, verbosity=0)
    m_default = fit(
        EvoTreeRegressor(loss=:mse, nrounds=50, early_stopping_rounds=5, eta=0.05, seed=123);
        x_train=x_train_es,
        y_train=y_train_es,
        x_eval=x_eval_es,
        y_eval=y_eval_es,
        verbosity=0,
    )

    @test length(m_tolerant.trees) < length(m_strict.trees)
    @test length(m_strict.trees) == params_strict.nrounds + 1
    @test m_tolerant.info[:logger][:iter_since_best] >= params_tolerant.early_stopping_rounds
    @test m_tolerant.info[:logger][:best_iter] < m_strict.info[:logger][:best_iter]
    @test length(m_default.trees) == length(m_strict.trees)
    @test predict(m_default, x_eval_es) == predict(m_strict, x_eval_es)
end

@testset "EvoTreeRegressor - logloss" begin
    params1 = EvoTreeRegressor(
        loss=:logloss,
        nrounds=100,
        lambda=0.5,
        gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Gamma" begin
    params1 = EvoTreeRegressor(
        loss=:gamma,
        nrounds=100,
        lambda=0.5,
        # gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Tweedie" begin
    params1 = EvoTreeRegressor(
        loss=:tweedie,
        nrounds=100,
        lambda=0.5,
        gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - L1" begin
    params1 = EvoTreeRegressor(
        loss=:mae,
        alpha=0.5,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.0,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Quantile" begin
    params1 = EvoTreeRegressor(
        loss=:quantile,
        alpha=0.5,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.0,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - MultiQuantile" begin
    alphas = [0.2, 0.5, 0.8]
    params1 = EvoTreeRegressor(
        loss=:multiquantile,
        alphas=alphas,
        nrounds=50,
        nbins=16,
        lambda=0.5,
        gamma=0.0,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    @test size(preds_ini) == (length(y_eval), length(alphas))

    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    @test size(preds) == (length(y_eval), length(alphas))
    @test !any(isnan.(preds))
end

@testset "EvoTreeCount - Count" begin
    params1 = EvoTreeCount(
        nrounds=100,
        lambda=0.5,
        gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeMLE" begin
    mle_kwargs = (
        nrounds=100,
        nbins=16,
        lambda=0.0,
        gamma=0.0,
        eta=0.1,
        max_depth=6,
        min_weight=10.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )
    configs = [
        EvoTreeMLE(; loss=:gaussian_mle, mle_kwargs...),
        EvoTreeMLE(; loss=:logistic_mle, mle_kwargs...),
        EvoTreeGaussian(; mle_kwargs...),
    ]
    @testset "$(nameof(typeof(params1))) $(params1.loss)" for params1 in configs
        model, cache = EvoTrees.init(params1, x_train, y_train)
        preds_ini = EvoTrees.predict(model, x_eval)
        @test size(preds_ini, 2) == 2
        @test all(>(0), preds_ini[:, 2])
        mse_error_ini = mean(abs.(preds_ini[:, 1] .- y_eval) .^ 2)
        model = fit(
            params1;
            x_train,
            y_train,
            x_eval,
            y_eval,
            print_every_n=100
        )

        preds = EvoTrees.predict(model, x_eval)
        @test all(>(0), preds[:, 2])
        mse_error = mean(abs.(preds[:, 1] .- y_eval) .^ 2)
        mse_gain_pct = mse_error / mse_error_ini - 1
        @test mse_gain_pct < -0.75
    end
end

@testset "EvoTrees - Feature Importance" begin
    params1 = EvoTreeRegressor(
        loss=:mse,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.1,
        eta=0.1,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        seed=123,
    )

    model = fit(params1; x_train, y_train)
    features_gain = EvoTrees.importance(model)
end


@testset "EvoTreeClassifier" begin
    x_train = Array([
        sin.(1:1000) rand(1000)
        100 .* cos.(1:1000) rand(1000).+1
    ])
    y_train = repeat(1:2; inner=1000)

    seed = 123
    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, seed)
    model = fit(params1; x_train, y_train)

    preds = EvoTrees.predict(model, x_train)[:, 1]
    @test !any(isnan.(preds))

    # Categorical array
    y_train_cat = CategoricalArray(y_train; levels=1:2)

    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, seed)
    model_cat = fit(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat ≈ preds

    # Categorical array with additional levels
    y_train_cat = CategoricalArray(y_train; levels=1:3)

    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, seed)
    model_cat = fit(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat ≈ preds # differences due to different stream of random numbers
end


@testset "check_args functionality" begin
    # check_args should throw an exception if the parameters are invalid
    @testset "check_parameter" begin
        # Valid case tests
        @test check_parameter(Float64, 1.5, 0.0, typemax(Float64), :lambda) === nothing
        @test check_parameter(Int, 5, 1, typemax(Int), :nrounds) === nothing
        @test check_parameter(Int, 1, 1, typemax(Int), :nrounds) === nothing
        @test check_parameter(Int, 1, 1, 1, :nrounds) === nothing

        # Invalid type tests
        @test_throws ErrorException check_parameter(Int, 1.5, 0, typemax(Int), :nrounds)
        @test_throws ErrorException check_parameter(Float64, "1.5", 0.0, typemax(Float64), :lambda)

        # Out of range tests
        @test_throws ErrorException check_parameter(Int, -5, 0, typemax(Int), :nrounds)
        @test_throws ErrorException check_parameter(Float64, -0.1, 0.0, typemax(Float64), :lambda)
        @test_throws ErrorException check_parameter(Int, typemax(Int64), 0, typemax(Int) - 1, :nrounds)
        @test_throws ErrorException check_parameter(Float64, typemax(Float64), 0.0, 10^6, :lambda)
    end

    # Check the implemented parameters on construction
    @testset "check_args all for EvoTreeRegressor" begin
        for (key, vals_to_test) in zip(
            [:nrounds, :max_depth, :nbins, :lambda, :gamma, :min_weight, :alpha, :rowsample, :colsample, :eta,
                :L2, :bagging_size, :early_stopping_rounds],
            [[-1, 1.5], [0, 1.5], [1, 256, 100.5], [-eps(Float64)], [-eps(Float64)], [-eps(Float64)],
                [-0.1, 1.1], [0.0f0, 1.1f0], [0.0, 1.1], [-eps(Float64)],
                [-eps(Float64)], [0, -1, 1.5], [-1, 1.5]])
            for val in vals_to_test
                @test_throws Exception EvoTreeRegressor(; zip([key], [val])...)
            end
        end
    end

    @testset "check_args EvoTreeRegressor - MultiQuantile" begin
        @test_throws Exception EvoTreeRegressor(loss=:multiquantile, alphas=[0.6, 0.4])
        @test_throws Exception EvoTreeRegressor(loss=:multiquantile, alphas=[0.0, 0.5])
        @test_throws Exception EvoTreeRegressor(loss=:multiquantile, alphas=[0.5, 0.5])
        config = EvoTreeRegressor(loss=:multiquantile, alphas=[0.5], nrounds=1)
        @test check_args(config) === nothing
    end

    @testset "check_args L2 and bagging_size" begin
        # Both used to be accepted unvalidated. A negative `L2` lands in the leaf denominator
        # and takes every prediction to NaN; a `bagging_size` below 1 makes the per-round loop
        # an empty range, so no trees are grown and the fit reports success anyway.
        seed!(7)
        x = rand(200, 3)
        y = 2 .* x[:, 1] .+ 0.2 .* randn(200)

        @test_throws Exception EvoTreeRegressor(L2=-100.0)
        @test_throws Exception EvoTreeRegressor(bagging_size=0)
        @test_throws Exception EvoTreeRegressor(early_stopping_rounds=-1)

        # `L2` is the sibling of `lambda`, which was already bounded below at zero.
        @test EvoTreeRegressor(L2=0.0) isa EvoTreeRegressor
        @test EvoTreeRegressor(bagging_size=1) isa EvoTreeRegressor
        @test EvoTreeRegressor(early_stopping_rounds=0) isa EvoTreeRegressor

        # Valid values still train, and every round still grows a tree.
        m = fit(EvoTreeRegressor(nrounds=10, max_depth=3, bagging_size=2, L2=1.0);
            x_train=x, y_train=y, verbosity=0)
        @test length(m.trees) - 1 == 20
        @test all(isfinite, predict(m, x))

        # Mutating a fitted config is the path MLJ tuning takes, so the second `check_args`
        # method must reject the same values.
        config = EvoTreeRegressor(nrounds=5)
        config.L2 = -1.0
        @test_throws Exception check_args(config)
        config = EvoTreeRegressor(nrounds=5)
        config.bagging_size = 0
        @test_throws Exception check_args(config)
    end

    # Test all EvoTypes that they have *some* checks in place
    @testset "check_args EvoTypes" begin
        for EvoTreeType in [EvoTreeMLE, EvoTreeGaussian, EvoTreeCount, EvoTreeClassifier, EvoTreeRegressor]
            config = EvoTreeType(nbins=32)
            # should not throw an exception
            @test check_args(config) === nothing
            # invalid nbins
            config.nbins = 256
            @test_throws Exception check_args(config)
        end
    end

    @testset "classifier target levels" begin
        rng = Xoshiro(7)
        x = rand(rng, 40, 3)

        for (y, k) in ((repeat(["a", "b", "c"], 20)[1:40], 3), (repeat(["a", "b"], 20), 2))
            m = fit(EvoTreeClassifier(nrounds=3); x_train=x, y_train=y)
            @test m.K == k
            @test size(predict(m, x)) == (40, k)
        end

        # A single-level target is not a meaningful classification problem, and is
        # rejected at fit rather than producing a degenerate model.
        @test_throws ErrorException fit(
            EvoTreeClassifier(nrounds=3); x_train=x, y_train=fill("a", 40)
        )
        @test_throws ErrorException fit(
            EvoTreeClassifier(nrounds=3); x_train=x,
            y_train=categorical(fill("a", 40), levels=["a"]),
        )

        # An unsupported target eltype used to be reported with `@error`, which logs but
        # does not throw, so execution continued to `length(target_levels)` with
        # `target_levels` still `nothing` and the user saw a `MethodError` instead.
        @test_throws ErrorException fit(
            EvoTreeClassifier(nrounds=3); x_train=x, y_train=rand(40)
        )
    end

    @testset "empty subsample" begin
        rng = Xoshiro(11)
        x = rand(rng, 40, 3)
        y = 2 .* x[:, 1] .+ 0.2 .* randn(rng, 40)

        # `cond` is `round(UInt8, 255 * rowsample)`, so a rowsample below 0.00196 keeps
        # only mask bytes equal to zero, roughly 1 in 256 rows. On small data the draw
        # comes up empty, which must be reported rather than silently producing a model
        # that predicts the bias for every row.
        @test_throws ErrorException fit(
            EvoTreeRegressor(nrounds=5, max_depth=3, rowsample=0.001, min_weight=0.0, seed=1);
            x_train=x, y_train=y, verbosity=0
        )

        # A normal rowsample is unaffected.
        m = fit(
            EvoTreeRegressor(nrounds=10, max_depth=3, rowsample=0.5, seed=1);
            x_train=x, y_train=y, verbosity=0
        )
        @test all(isfinite, predict(m, x))
    end

    @testset "importance with no splits" begin
        rng = Xoshiro(11)
        x = rand(rng, 300, 4)
        y = rand(rng, 300)

        # A model with no split anywhere has zero total gain. Normalising by that total
        # would give NaN for every feature, so the zeros are returned as they are.
        for config in (
            EvoTreeRegressor(nrounds=5, max_depth=1),
            EvoTreeRegressor(nrounds=0),
            EvoTreeRegressor(nrounds=5, max_depth=4, gamma=1e9),
        )
            m = fit(config; x_train=x, y_train=y)
            imp = EvoTrees.importance(m)
            @test !any(isnan(v) for (_, v) in imp)
            @test all(v == 0 for (_, v) in imp)
            @test length(imp) == 4
        end

        # A model that does split still normalises to one.
        m = fit(EvoTreeRegressor(nrounds=20, max_depth=4); x_train=x, y_train=y)
        imp = EvoTrees.importance(m)
        @test sum(v for (_, v) in imp) ≈ 1
        @test issorted([v for (_, v) in imp]; rev=true)

        # Ordering is stable across calls, and duplicate names are not collapsed.
        @test first.(EvoTrees.importance(m)) == first.(EvoTrees.importance(m))
        @test length(EvoTrees.importance(m; feature_names=[:a, :a, :b, :c])) == 4
    end

    @testset "predict with mismatched column count" begin
        rng = Xoshiro(5)
        x = rand(rng, 300, 4)
        y = x[:, 1] .+ 2 .* x[:, 4]
        m = fit(EvoTreeRegressor(nrounds=10, max_depth=4); x_train=x, y_train=y)

        @test EvoTrees.predict(m, x) == EvoTrees.predict(m, x)
        for k in (1, 2, 3)
            @test_throws ErrorException EvoTrees.predict(m, x[:, 1:k])
        end
        @test_throws ErrorException EvoTrees.predict(m, hcat(x, rand(rng, 300, 2)))
    end

    @testset "gamma target support" begin
        rng = Xoshiro(21)
        x = rand(rng, 200, 3)
        ypos = 1.0 .+ rand(rng, 200)

        m = fit(EvoTreeRegressor(loss=:gamma, nrounds=5); x_train=x, y_train=ypos)
        @test length(m.trees) == 6

        for bad in (0.0, -1.0)
            y = copy(ypos)
            y[7] = bad
            @test_throws ErrorException fit(
                EvoTreeRegressor(loss=:gamma, nrounds=5); x_train=x, y_train=y
            )
        end

        ymat = permutedims(hcat(ypos, ypos .+ 1))
        m = fit(EvoTreeRegressor(loss=:gamma, nrounds=5); x_train=x, y_train=ymat)
        @test length(m.trees) == 6
        ymat[2, 5] = 0.0
        @test_throws ErrorException fit(
            EvoTreeRegressor(loss=:gamma, nrounds=5); x_train=x, y_train=ymat
        )

        yzero = copy(ypos)
        yzero[7] = 0.0
        for loss in (:poisson, :tweedie)
            m = fit(EvoTreeRegressor(loss=loss, nrounds=5); x_train=x, y_train=yzero)
            @test length(m.trees) == 6
        end
    end

end
