using Tables
using MLJ
using StatsBase: sample
using Revise
using EvoTrees
using EvoTrees: logit, sigmoid, predict
import EvoTrees: EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian

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
tree_model = EvoTreeRegressor(max_depth=5, η=0.01, nrounds=10)
tree = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
MLJ.fit!(tree, rows=train, verbosity=1)

tree.model.nrounds += 10
@time MLJ.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))
mean(abs.(pred_train - MLJ.selectrows(Y,train)))

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_test = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(pred_test - MLJ.selectrows(Y,test)))


##################################################
### classif
##################################################
features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Int.(round.(Y)) .+ 1
y = string.(y)
y = CategoricalArray(y, ordered=false)
X = Tables.table(X)

# @load EvoTreeRegressor
tree_model = EvoTreeClassifier(max_depth=5, η=0.01, λ=0.0, γ=0.0, nrounds=10, K=2)
tree = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
MLJ.fit!(tree, rows=train, verbosity=1)

tree.model.nrounds += 10
@time MLJ.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))

y_levels = classes(y[1])
pred_mlj = [MLJBase.UnivariateFinite(y_levels, pred_train[i,:]) for i in 1:size(pred_train, 1)]
cross_entropy(pred_mlj, MLJ.selectrows(y,train))

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
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

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
@time tree = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time MLJ.fit!(tree, rows=train, verbosity=1, force=true)

tree.model.nrounds = 10
tree.cache.params.nrounds = 10

using MLJBase
tree.model.nrounds += 10
@time EvoTrees.grow_gbtree_MLJ!(tree.fitresult, tree.cache, verbosity=1)

tree.model.nrounds += 10
@time MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, Y)

tree.model.nrounds += 1
@time MLJ.fit!(tree, rows=train, verbosity=0)

@time for i in 1:10
    tree.model.nrounds += 1
    MLJ.fit!(tree, rows=train, verbosity=0)
end

@time for i in 1:1
    tree.model.nrounds += 10
    MLJ.fit!(tree, rows=train, verbosity=0)
end

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_train = MLJ.predict(tree, MLJ.selectrows(X,train))
mean(abs.(pred_train - MLJ.selectrows(Y,train)))
