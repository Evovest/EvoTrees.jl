using Tables
using MLJ
using MLJBase
using StatsBase: sample
using CategoricalArrays
using Distributions
using Revise
using EvoTrees
using EvoTrees: logit, sigmoid
import EvoTrees: EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian

##################################################
### Regrtession - small data
##################################################
features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = Tables.table(X)

# @load EvoTreeRegressor
# linear regression
tree_model = EvoTreeRegressor(max_depth=5, η=0.05, nrounds=10)
# logistic regression
tree_model = EvoTreeRegressor(loss=:logistic, max_depth=5, η=0.05, nrounds=10)
# quantile regression
tree_model = EvoTreeRegressor(loss=:quantile, α=0.75, max_depth=5, η=0.05, nrounds=10)

tree = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
MLJ.fit!(tree, rows=train, verbosity=1)

tree.model.nrounds += 10
MLJ.fit!(tree, rows=train, verbosity=1)

# predict on train data
pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))
mean(pred_train)
mean(abs.(pred_train - MLJ.selectrows(Y,train)))

# predict on test data
pred_test = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(pred_test - MLJ.selectrows(Y,test)))


##################################################
### classif
##################################################
X, y = @load_crabs

# features = rand(10_000) .* 5 .- 2
# X = reshape(features, (size(features)[1], 1))
# Y = sin.(features) .* 0.5 .+ 0.5
# Y = logit(Y) + randn(size(Y))
# Y = sigmoid(Y)
# y = Int.(round.(Y)) .+ 1
# y = _levels[y]
# # y = string.(y)
# y = CategoricalArray(y, ordered=false)
# X = Tables.table(X)
# X_matrix = MLJBase.matrix(X)

# define hyperparameters
tree_model = EvoTreeClassifier(max_depth=5, η=0.01, λ=0.0, γ=0.0, nrounds=10)

# @load EvoTreeRegressor
tree = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
MLJ.fit!(tree, rows=train, verbosity=1)

tree.model.nrounds += 10
MLJ.fit!(tree, rows=train, verbosity=1)

pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))
cross_entropy(pred_train, MLJ.selectrows(y, train)) |> mean
pred_train_mode = MLJ.predict_mode(tree, MLJ.selectrows(X,train))

pred_test = MLJ.predict(tree, MLJ.selectrows(X,test))
cross_entropy(pred_test, MLJ.selectrows(y, test)) |> mean
pred_test_mode = MLJ.predict_mode(tree, MLJ.selectrows(X,test))

##################################################
### regression - Larger data
##################################################
features = rand(100_000, 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeRegressor(
    loss=:linear, metric=:mae,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

X = Tables.table(X)
X = Tables.rowtable(X)
X = Tables.columntable(X)
X_matrix = MLJBase.matrix(X)

# typeof(X)
@time tree = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time MLJ.fit!(tree, rows=train, verbosity=1, force=true)
@time test_fit!(tree, rows=train, verbosity=1, force=true)
@time EvoTrees.grow_gbtree_MLJ(X_matrix, Y, tree_model, verbosity=1)
@time EvoTrees.grow_gbtree(X_matrix, Y, tree_model, verbosity=1)

tree.model.nrounds = 10
tree.cache.params.nrounds = 10

tree.model.nrounds += 10
@time MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, Y)
# @time x1, x2, x3 = MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, Y)

tree.model.nrounds += 10
@time MLJ.fit!(tree, rows=train, verbosity=1)
# @time MLJBase.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))
mean(abs.(pred_train - MLJ.selectrows(Y,train)))
