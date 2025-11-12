using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
using ShapML
using Plots

df = MLDatasets.BostonHousing().dataframe
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, Not(:MEDV)]), train_data[:, :MEDV]
x_eval, y_eval = Matrix(eval_data[:, Not(:MEDV)]), eval_data[:, :MEDV]

config = EvoTreeRegressor(
    nrounds=200,
    eta=0.1,
    max_depth=5,
    lambda=0.1,
    rowsample=1.0,
    colsample=1.0,
    early_stopping_rounds=10,
    nbins=16
)

m = EvoTrees.fit(config;
    x_train, y_train,
    x_eval, y_eval,
    print_every_n=10)

EvoTrees.importance(m)
plot(m, 2)
m.trees[2].feat
Int.(m.trees[2].cond_bin)

# inconsistent result in EvoTrees.importance and ShapML shap values
tshap1 = EvoTrees.treeshapv1(m, x_train[1:1, :])
tshap1 = EvoTrees.treeshapv1(m, x_train[4:4, :])

# FIXME: following observation crashed:
# BoundsError: attempt to access 0-element Vector{Float32} at index [0]
# length(w) is 0, as an input of `unwind`, resulting in a crash at line 145: n = w[lw+1]
tshap1 = EvoTrees.treeshapv1(m, x_train[2:2, :])

function shap_pred_fun(m, data)
    p = m(data)
    return p
end
explain = DataFrame(x_train[1:1, :], m.info[:feature_names])
reference = DataFrame(x_train, m.info[:feature_names])
shap_pred_fun(m, explain)
shap_iter = ShapML.shap(; explain, reference, model=m, predict_function=shap_pred_fun, sample_size=64, seed=123)
sort!(shap_iter, :shap_effect)
