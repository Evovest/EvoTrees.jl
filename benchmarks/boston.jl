
using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

df = MLDatasets.BostonHousing().dataframe
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, Not(:MEDV)]), train_data[:, :MEDV]
x_eval, y_eval = Matrix(eval_data[:, Not(:MEDV)]), eval_data[:, :MEDV]

config = EvoTreeRegressor(
    loss=:mse,
    metric=:mse,
    nrounds=1,
    early_stopping_rounds=10,
    eta=0.1,
    max_depth=2,
    lambda=0.0,
    L2=0.0,
    rowsample=0.9,
    colsample=0.9)

model_mse = EvoTrees.fit(config;
    x_train, y_train,
    x_eval, y_eval,
    print_every_n=1)

pred_train = model(x_train)
pred_eval = model(x_eval)

mean(abs.(pred_train .- y_train))
mean(abs.(pred_eval .- y_eval))

