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
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:mse,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - logloss" begin
    params1 = EvoTreeRegressor(
        loss=:logloss,
        nrounds=100,
        lambda=0.5,
        gamma=0.1,
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:logloss,
        print_every_n=25
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
        gamma=0.1,
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:gamma,
        print_every_n=25
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
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:tweedie,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - L1" begin
    params1 = EvoTreeRegressor(
        loss=:l1,
        alpha=0.5,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:mae,
        print_every_n=25
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
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:wmae,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeCount - Count" begin
    params1 = EvoTreeCount(
        loss=:poisson,
        nrounds=100,
        lambda=0.5,
        gamma=0.1,
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:poisson_deviance,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeMLE - Gaussian" begin
    params1 = EvoTreeMLE(
        loss=:gaussian,
        nrounds=100,
        nbins=16,
        lambda=0.0,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=10.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:gaussian,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeMLE - Logistic" begin
    params1 = EvoTreeMLE(
        loss=:logistic,
        nrounds=100,
        nbins=16,
        lambda=0.0,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=10.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric=:logistic_mle,
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeGaussian - Gaussian" begin
    params1 = EvoTreeGaussian(
        nrounds=100,
        nbins=16,
        lambda=0.0,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=10.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model, cache = EvoTrees.init(params1, x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric="gaussian_mle",
        print_every_n=25
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTrees - Feature Importance" begin
    params1 = EvoTreeRegressor(
        loss=:mse,
        nrounds=100,
        nbins=16,
        lambda=0.5,
        gamma=0.1,
        eta=0.05,
        max_depth=6,
        min_weight=1.0,
        rowsample=0.5,
        colsample=1.0,
        rng=123,
    )

    model = fit_evotree(params1; x_train, y_train)
    features_gain = EvoTrees.importance(model)
end


@testset "EvoTreeClassifier" begin
    x_train = Array([
        sin.(1:1000) rand(1000)
        100 .* cos.(1:1000) rand(1000).+1
    ])
    y_train = repeat(1:2; inner=1000)

    rng = rand(UInt32)
    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, rng)
    model = fit_evotree(params1; x_train, y_train)

    preds = EvoTrees.predict(model, x_train)[:, 1]
    @test !any(isnan.(preds))

    # Categorical array
    y_train_cat = CategoricalArray(y_train; levels=1:2)

    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, rng)
    model_cat = fit_evotree(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat ≈ preds

    # Categorical array with additional levels
    y_train_cat = CategoricalArray(y_train; levels=1:3)

    params1 = EvoTreeClassifier(; nrounds=100, eta=0.3, rng)
    model_cat = fit_evotree(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat ≈ preds # differences due to different stream of random numbers
end

@testset "Parametric kwarg constructor" begin

    @testset "_type2loss" begin
        # utility that converts types into loss symbols for EvoTreeRegressor
        @test EvoTrees._type2loss(EvoTrees.MSE) == :mse
        @test EvoTrees._type2loss(EvoTrees.L1) == :l1
        @test EvoTrees._type2loss(EvoTrees.LogLoss) == :logloss
        @test EvoTrees._type2loss(EvoTrees.Gamma) == :gamma
        @test EvoTrees._type2loss(EvoTrees.Tweedie) == :tweedie
        @test EvoTrees._type2loss(EvoTrees.Quantile) == :quantile
    end

    # check if we retain the parametric information properly
    for EvoParamType in [
        EvoTreeRegressor{EvoTrees.MSE},
        EvoTreeRegressor{EvoTrees.L1},
        EvoTreeCount{EvoTrees.Poisson},
        EvoTreeClassifier{EvoTrees.MLogLoss},
        EvoTreeMLE{EvoTrees.LogisticMLE},
        EvoTreeGaussian{EvoTrees.GaussianMLE}
    ]

        config = EvoParamType(; max_depth=2)
        @test config isa EvoParamType
        @test config.max_depth == 2
    end
end


@testset "check_args functionality" begin
    # check_args should throw an exception if the parameters are invalid
    @testset "check_parameter" begin
        # Valid case tests
        @test check_parameter(Float64, 1.5, 0.0, typemax(Float64), :lambda) == nothing
        @test check_parameter(Int, 5, 1, typemax(Int), :nrounds) == nothing
        @test check_parameter(Int, 1, 1, typemax(Int), :nrounds) == nothing
        @test check_parameter(Int, 1, 1, 1, :nrounds) == nothing

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
            [:nrounds, :max_depth, :nbins, :lambda, :gamma, :min_weight, :alpha, :rowsample, :colsample, :eta],
            [[-1, 1.5], [0, 1.5], [1, 256, 100.5], [-eps(Float64)], [-eps(Float64)], [-eps(Float64)],
                [-0.1, 1.1], [0.0f0, 1.1f0], [0.0, 1.1], [-eps(Float64)]])
            for val in vals_to_test
                @test_throws Exception EvoTreeRegressor(; zip([key], [val])...)
            end
        end
    end

    # Test all EvoTypes that they have *some* checks in place
    @testset "check_args EvoTypes" begin
        for EvoTreeType in [EvoTreeMLE, EvoTreeGaussian, EvoTreeCount, EvoTreeClassifier, EvoTreeRegressor]
            config = EvoTreeType(nbins=32)
            # should not throw an exception
            @test check_args(config) == nothing
            # invalid nbins
            config.nbins = 256
            @test_throws Exception check_args(config)
        end
    end

end