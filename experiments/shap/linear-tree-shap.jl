using EvoTrees
using EvoTrees: Shap
using Statistics
using Random: seed!

######################
# Minimal checks
######################
seed!(123)
x_bin = UInt8.(rand(1:5, 10, 5))
tree = EvoTrees.Tree{EvoTrees.MSE,1}(2)
tree.split[1] = true
tree.feat[1] = 1
tree.cond_bin[1] = 3
tree.pred[1, 2] = 11.0
tree.pred[1, 3] = 13.0
tree.w .= [4.0, 3.0, 1.0]

stree = Shap.ShapTree(tree)
shap = Shap.inference(stree, x_bin)

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

# x_train = x_train[:, 1:3]
config = EvoTreeRegressor(
    loss=:mse,
    metric=:mse,
    nrounds=100,
    early_stopping_rounds=10,
    eta=0.05,
    max_depth=5,
    lambda=0.0,
    L2=0.0,
    rowsample=1.0,
    colsample=1.0)

m = EvoTrees.fit(config; x_train, y_train)
m.trees[2];

EvoTrees.importance(m)
p_full = m(x_train)
p_tree = p_full .- mean(p_full)

x_bin = EvoTrees.binarize(x_train; feature_names=m.info[:feature_names], edges=m.info[:edges])
@time tree = Shap.ShapTree(m.trees[2]);
@time _shap = Shap.inference(tree, x_bin);
@time shap = EvoTrees.shap(m, x_train)

# # obs decomposition
sum(shap[1, :])
sum(shap[2, :])
sum(shap[3, :])

# feat importance
tree_shap_imp = mean(abs.(shap); dims=1)
tree_shap_imp ./= sum(tree_shap_imp)

# ShapML reconciliation
function shap_pred_fun(m, data)
    p = m(data)
    return p
end
explain = DataFrame(x_train[1:end, :], m.info[:feature_names])
reference = DataFrame(x_train, m.info[:feature_names])
shap_pred_fun(m, explain)
@time shap_ml = ShapML.shap(; explain, reference, model=m, predict_function=shap_pred_fun, sample_size=64, seed=123)
shapml_imp = combine(groupby(shap_ml, :feature_name), :shap_effect => (x -> mean(abs.(x))) => :importance)
transform!(shapml_imp, :importance => (x -> x ./ sum(x)) => :importance)
shap_ml[shap_ml.index.==1, :shap_effect]
shap[1, :]
