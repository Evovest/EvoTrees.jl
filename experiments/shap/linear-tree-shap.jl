using EvoTrees
using EvoTrees.LinearTreeShap
using Statistics

# small random dataset
x_bin = UInt8.(rand(1:5, 10, 5))
tree = EvoTrees.Tree{EvoTrees.MSE,1}(2)
tree.split[1] = true
tree.feat[1] = 1
tree.cond_bin[1] = 3
tree.pred[1, 2] = -1.0
tree.pred[1, 3] = 1.0
tree.w .= [4.0, 3.0, 1.0]

ltree = LinearTreeShap.copy_tree(tree)
S = LinearTreeShap.inference(ltree, x_bin; debug=true)
println("SHAP matrix size: ", size(S))

nobs = 100_000
# x_train = randn(nobs, 2)
x_train = rand(Bool, nobs, 2)
y_train = 1.0 .* x_train[:, 1] .+ 0.5 * x_train[:, 2] .+ randn(nobs) .* 0.01
learner = EvoTreeRegressor(nrounds=1, eta=0.5, max_depth=5)
m = EvoTrees.fit(learner; x_train, y_train)
x_bin = EvoTrees.binarize(x_train; feature_names=m.info[:feature_names], edges=m.info[:edges])

shap = zeros(size(x_bin))
for tree in m.trees[2:end]
    ltree = LinearTreeShap.copy_tree(tree)
    shap .+= LinearTreeShap.inference(ltree, x_bin; debug=false)
end
mean(abs.(shap); dims=1)

ltree = LinearTreeShap.copy_tree(m.trees[2])
shap = LinearTreeShap.inference(ltree, x_bin; debug=false)
mean(abs.(shap); dims=1)


######################
# Boston
######################
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
import ShapML

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
    eta=1.0,
    max_depth=3,
    lambda=0.0,
    L2=0.0,
    rowsample=1.0,
    colsample=1.0)

m = EvoTrees.fit(config;
    x_train, y_train,
    x_eval, y_eval,
    print_every_n=1)

EvoTrees.importance(m)

p_full = m(x_train)
p_tree = p_full .- mean(p_full)

ltree = LinearTreeShap.copy_tree(m.trees[2])
x_bin = EvoTrees.binarize(x_train; feature_names=m.info[:feature_names], edges=m.info[:edges])
shap = LinearTreeShap.inference(ltree, x_bin; debug=false)

# feat importance
tree_shap_imp = mean(abs.(shap); dims=1)
tree_shap_imp ./ sum(tree_shap_imp)

# obs decomposition
sum(shap[2, :])
sum(shap[3, :])
# m.trees[2].pred
# proper spread betwee idx 2 and 3, but not overall level...
sum(shap[3, :]) - sum(shap[2, :])
p_tree[3] - p_tree[2]

function shap_pred_fun(m, data)
    p = m(data)
    return p
end
explain = DataFrame(x_train[1:1, :], m.info[:feature_names])
reference = DataFrame(x_train, m.info[:feature_names])
shap_pred_fun(m, explain)
shap_iter = ShapML.shap(; explain, reference, model=m, predict_function=shap_pred_fun, sample_size=64, seed=123)
sort!(shap_iter, :shap_effect)