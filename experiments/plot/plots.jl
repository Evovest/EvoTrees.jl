using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
using Plots

df = MLDatasets.Titanic().dataframe
Random.seed!(123)

transform!(df, :Sex => categorical => :Sex)
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]
dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]
target_name = "Survived"
feature_names = setdiff(names(df), [target_name])

config = EvoTreeRegressor(
    loss=:logloss,
    nrounds=200,
    early_stopping_rounds=10,
    eta=0.05,
    nbins=16,
    max_depth=4,
    rowsample=0.5,
    colsample=0.9)

model = EvoTrees.fit(
    config, dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10)

p = plot(model, 2)
# plot(model)

savefig(p, "docs/src/assets/plot_tree.png")
