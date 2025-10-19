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

x_num_m1 = Vector{Union{Missing,Float64}}(copy(x_num))
x_num_m2 = Vector{Any}(copy(x_num))
lvls_m1 = ["a", "b", "c", missing]
x_cat_m1 = categorical(rand(lvls_m1, nobs), levels=lvls)
x_bool_m1 = Vector{Union{Missing,Bool}}(copy(x_bool))

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
y_tot = sigmoid(y_tot)
y_tot_m1 = allowmissing(y_tot)
y_tot_m1[1] = missing

config = EvoTreeRegressor(
    loss=:mse,
    nrounds=100,
    nbins=16,
    lambda=0.5,
    gamma=0.1,
    eta=0.1,
    max_depth=3,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    seed=123,
)

@testset "DataFrames - missing features" begin

    df_tot = DataFrame(x_num=x_num, x_bool=x_bool, x_cat=x_cat, y=y_tot)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]

    model = fit(
        config,
        dtrain;
        target_name)

    @test model.info[:feature_names] == [:x_num, :x_bool, :x_cat]

    # keep only feature_names <= Real or Categorical
    df_tot = DataFrame(x_num=x_num, x_num_m1=x_num_m1, x_num_m2=x_num_m2,
        x_cat_m1=x_cat_m1, x_bool_m1=x_bool_m1, y=y_tot)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]

    model = fit(
        config,
        dtrain;
        target_name,
        deval)

    @test model.info[:feature_names] == [:x_num]

    model = fit(
        config,
        dtrain;
        target_name,
        feature_names=[:x_num])

    @test model.info[:feature_names] == [:x_num]

    # specifyin features with missings should error
    @test_throws AssertionError fit(
        config,
        dtrain;
        deval,
        feature_names=[:x_num, :x_num_m1, :x_num_m2, :x_cat_m1, :x_bool_m1],
        target_name)

end

@testset "DataFrames - missing in target errors" begin

    df_tot = DataFrame(x_num=x_num, x_bool=x_bool, x_cat=x_cat, y=y_tot_m1)
    dtrain, deval = df_tot[i_train, :], df_tot[i_eval, :]

    @test_throws AssertionError fit(
        config,
        dtrain;
        target_name)

end

@testset "Matrix - missing features" begin

    x_tot = allowmissing(hcat(x_num_m1))
    @test_throws AssertionError fit(
        config;
        x_train=x_tot,
        y_train=y_tot)

    x_tot = Matrix{Any}(hcat(x_num_m2))
    @test_throws AssertionError fit(
        config;
        x_train=x_tot,
        y_train=y_tot)

end

