using Statistics
using StatsBase: sample
using EvoTrees: sigmoid, logit
using EvoTrees: check_args, check_parameter
using CategoricalArrays
using DataFrames
using Random: seed!

# prepare a dataset
seed!(123)
nobs = 1_000
x_num = rand(nobs) .* 5
lvls = ["a", "b", "c"]
x_cat = categorical(rand(lvls, nobs), levels=lvls, ordered=false)
x_bool = rand(Bool, nobs)

# train-eval split
is = collect(1:nobs)
i_sample = sample(is, nobs, replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * nobs)]
i_eval = i_sample[floor(Int, train_size * nobs)+1:end]

# target var
y_tot = sin.(x_num) .* 0.5 .+ 0.5
y_tot = logit(y_tot) + randn(nobs)
y_tot = sigmoid(y_tot)
target_name = "y"

config = EvoTreeRegressor(
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

@testset "Tables - NTuples" begin

    dtrain = (x1=x_num[i_train], y=y_tot[i_train])
    deval = (x1=x_num[i_eval], y=y_tot[i_eval])
    y_train, y_eval = y_tot[i_train], y_tot[i_eval]

    m, cache = EvoTrees.init(config, dtrain; target_name="y")
    preds_ini = EvoTrees.predict(m, deval)
    mse_error_ini = mean((preds_ini .- y_eval) .^ 2)
    model = fit(
        config,
        dtrain;
        target_name)

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

    model = fit(
        config,
        dtrain;
        target_name,
        deval,
        print_every_n=25)

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

end

@testset "Tables - DataFrames" begin

    df_tot = DataFrame(x_num=x_num, y=y_tot)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]
    y_train, y_eval = y_tot[i_train], y_tot[i_eval]

    m, cache = EvoTrees.init(config, dtrain; target_name="y")
    preds_ini = EvoTrees.predict(m, deval)
    mse_error_ini = mean((preds_ini .- y_eval) .^ 2)
    model = fit(
        config,
        dtrain;
        target_name)

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

    model = fit(
        config,
        dtrain;
        target_name,
        deval,
        print_every_n=25)

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

end


@testset "Tables - num/bool/cat" begin

    y_tot = sin.(x_num) .* 0.5 .+ 0.5
    y_tot = logit(y_tot) .+ randn(nobs) .+ 1.0 .* (x_cat .== "b") .- 1.0 .* (x_cat .== "c") .+ 1.0 .* x_bool
    y_tot = sigmoid(y_tot)
    y_train, y_eval = y_tot[i_train], y_tot[i_eval]

    df_tot = DataFrame(x_num=x_num, x_bool=x_bool, x_cat=x_cat, y=y_tot)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]

    m, cache = EvoTrees.init(config, dtrain; target_name)
    preds_ini = EvoTrees.predict(m, deval)
    mse_error_ini = mean((preds_ini .- y_eval) .^ 2)
    model = fit(
        config,
        dtrain;
        target_name)

    @test model.info[:feature_names] == [:x_num, :x_bool, :x_cat]

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

end

@testset "Tables - bool/cat" begin

    y_tot = sin.(x_num) .* 0.1 .+ 0.5
    y_tot = logit(y_tot) .+ randn(nobs) .+ 2.0 .* (x_cat .== "b") .- 3.0 .* (x_cat .== "c") .+ 3.0 .* x_bool
    y_tot = sigmoid(y_tot)
    y_train, y_eval = y_tot[i_train], y_tot[i_eval]

    df_tot = DataFrame(x_num=x_num, x_bool=x_bool, x_cat=x_cat, y=y_tot)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]
    feature_names = [:x_bool, :x_cat]

    m, cache = EvoTrees.init(config, dtrain; target_name, feature_names)
    preds_ini = EvoTrees.predict(m, deval)
    mse_error_ini = mean((preds_ini .- y_eval) .^ 2)
    model = fit(
        config,
        dtrain;
        target_name,
        feature_names)

    @test model.info[:feature_names] == feature_names

    preds = EvoTrees.predict(model, deval)
    mse_error = mean((preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

end
