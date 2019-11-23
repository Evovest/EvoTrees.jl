using Tables
using MLJ
using EvoTrees
import EvoTrees: EvoTreeRegressor, predict
using EvoTrees: logit, sigmoid

features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = Tables.table(X)

@load EvoTreeRegressor
tree_model = EvoTreeRegressor(max_depth=5, η=0.01, nrounds=200)
tree = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
MLJ.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
yhat = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(yhat - MLJ.selectrows(Y,test)))

tree.fitresult
tree.model
up_res = EvoTrees.MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, y)

up_res = grow_gbtree!(tree.fitresult, MLJBase.matrix(X), y)
up_res
tree.fitresult

MLJ.fit!(tree, rows=train)
yhat = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(yhat - MLJ.selectrows(Y,test)))
