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
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

@testset "EvoTreeRegressor - Linear" begin
    # linear
    params1 = EvoTreeRegressor(
        loss=:linear, metric=:mse,
        nrounds=100, nbins=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Logistic" begin
    params1 = EvoTreeRegressor(
        loss=:logistic, metric=:logloss,
        nrounds=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Gamma" begin
    params1 = EvoTreeRegressor(
        loss=:gamma, metric=:gamma,
        nrounds=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Tweedie" begin
    params1 = EvoTreeRegressor(
        loss=:tweedie, metric=:tweedie,
        nrounds=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - L1" begin
    params1 = EvoTreeRegressor(
        loss=:L1, alpha=0.5, metric=:mae,
        nrounds=100, nbins=100,
        lambda=0.5, gamma=0.0, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeRegressor - Quantile" begin
    params1 = EvoTreeRegressor(
        loss=:quantile, alpha=0.5, metric=:quantile,
        nrounds=100, nbins=100,
        lambda=0.5, gamma=0.0, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeCount - Count" begin
    params1 = EvoTreeCount(
        loss=:poisson, metric=:poisson,
        nrounds=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTreeGaussian - Gaussian" begin
    params1 = EvoTreeGaussian(
        loss=:gaussian, metric=:gaussian,
        nrounds=100, nbins=100,
        lambda=0.0, gamma=0.0, eta=0.05,
        max_depth=6, min_weight=10.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model, cache = EvoTrees.init_evotree(params1, X_train, Y_train)
    preds_ini = EvoTrees.predict(model, X_eval)[:, 1]
    mse_error_ini = mean(abs.(preds_ini .- Y_eval) .^ 2)
    model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=25)

    preds = EvoTrees.predict(model, X_eval)[:, 1]
    mse_error = mean(abs.(preds .- Y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75
end

@testset "EvoTrees - Feature Importance" begin
    params1 = EvoTreeRegressor(
        loss=:linear, metric=:mae,
        nrounds=100, nbins=100,
        lambda=0.5, gamma=0.1, eta=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=1.0, rng=123)

    model = fit_evotree(params1, X_train, Y_train)
    features_gain = importance(model, 1:1)
end
