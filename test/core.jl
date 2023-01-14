using Statistics
using StatsBase: sample
using EvoTrees: sigmoid, logit
using Random: seed!

# prepare a dataset
seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace = false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

@testset "EvoTreeRegressor - Linear" begin
    # linear
    params1 = EvoTreeRegressor(
        loss = :linear,
        nrounds = 100,
        nbins = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :mse,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Logistic" begin
    params1 = EvoTreeRegressor(
        loss = :logistic,
        nrounds = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :logloss,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Gamma" begin
    params1 = EvoTreeRegressor(
        loss = :gamma,
        nrounds = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :gamma,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Tweedie" begin
    params1 = EvoTreeRegressor(
        loss = :tweedie,
        nrounds = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :tweedie,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - L1" begin
    params1 = EvoTreeRegressor(
        loss = :L1,
        alpha = 0.5,
        nrounds = 100,
        nbins = 100,
        lambda = 0.5,
        gamma = 0.0,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :mae,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Quantile" begin
    params1 = EvoTreeRegressor(
        loss = :quantile,
        alpha = 0.5,
        nrounds = 100,
        nbins = 100,
        lambda = 0.5,
        gamma = 0.0,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :wmae,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeCount - Count" begin
    params1 = EvoTreeCount(
        loss = :poisson,
        nrounds = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :poisson_deviance,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeMLE - Gaussian" begin
    params1 = EvoTreeMLE(
        loss = :gaussian,
        nrounds = 100,
        nbins = 100,
        lambda = 0.0,
        gamma = 0.0,
        eta = 0.05,
        max_depth = 6,
        min_weight = 10.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :gaussian,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeMLE - Logistic" begin
    params1 = EvoTreeMLE(
        loss = :logistic,
        nrounds = 100,
        nbins = 100,
        lambda = 0.0,
        gamma = 0.0,
        eta = 0.05,
        max_depth = 6,
        min_weight = 10.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = :logistic_mle,
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeGaussian - Gaussian" begin
    params1 = EvoTreeGaussian(
        nrounds = 100,
        nbins = 100,
        lambda = 0.0,
        gamma = 0.0,
        eta = 0.05,
        max_depth = 6,
        min_weight = 10.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model, cache = EvoTrees.init_evotree(params1; x_train, y_train)
    preds_ini = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit_evotree(
        params1;
        x_train,
        y_train,
        x_eval,
        y_eval,
        metric = "gaussian_mle",
        print_every_n = 25,
    )

    preds = EvoTrees.predict(model, x_eval)[:, 1]
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTrees - Feature Importance" begin
    params1 = EvoTreeRegressor(
        loss = :linear,
        nrounds = 100,
        nbins = 100,
        lambda = 0.5,
        gamma = 0.1,
        eta = 0.05,
        max_depth = 6,
        min_weight = 1.0,
        rowsample = 0.5,
        colsample = 1.0,
        rng = 123,
    )

    model = fit_evotree(params1; x_train, y_train)
    features_gain = importance(model)
end


@testset "EvoTreeClassifier" begin
    # x_train = Array([
    #     sin.(1:1000) cos.(1:1000)
    #     100 .* cos.(1:1000) 100 .* sin.(1:1000)
        
    # ])
    x_train = Array([
        sin.(1:1000) rand(1000)
        100 .* cos.(1:1000) rand(1000) .+ 1 
    ])
    y_train = repeat(1:2; inner = 1000)

    rng = rand(UInt32)
    params1 = EvoTreeClassifier(; T = Float32, nrounds = 100, eta = 0.3, rng)
    model = fit_evotree(params1; x_train, y_train)

    preds = EvoTrees.predict(model, x_train)[:, 1]
    @test !any(isnan.(preds))

    # Categorical array
    y_train_cat = CategoricalArray(y_train; levels=1:2)

    params1 = EvoTreeClassifier(; T = Float32, nrounds = 100, eta = 0.3, rng)
    model_cat = fit_evotree(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat == preds

    # Categorical array with additional levels
    y_train_cat = CategoricalArray(y_train; levels=1:3)

    params1 = EvoTreeClassifier(; T = Float32, nrounds = 100, eta = 0.3, rng)
    model_cat = fit_evotree(params1; x_train, y_train=y_train_cat)

    preds_cat = EvoTrees.predict(model_cat, x_train)[:, 1]
    @test preds_cat ≈ preds # differences due to different stream of random numbers
end
