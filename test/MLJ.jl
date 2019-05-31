using Tables
using MLJ
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
fit!(tree, rows=train)

yhat = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(yhat - MLJ.selectrows(Y,test)))

fit!(tree, rows=train)
yhat = MLJ.predict(tree, MLJ.selectrows(X,test))
mean(abs.(yhat - MLJ.selectrows(Y,test)))






# ABLAOM ADDED TO DEMO (CAN DELETE)

# see also
# https://github.com/alan-turing-institute/MLJ.jl/blob/master/examples/xgboost.jl

# tested using MLJ v2.2.0 or higher:

# no change, so no train:
fit!(tree, rows=train)

# increase nrounds triggers update (400 extra iterations):
tree_model.nrounds = 600
fit!(tree, rows=train)

# change any other hyperparameter except learning rate triggers
# retraining from scratch:
tree_model.λ = 0.01
fit!(tree, rows=train)

# can generate learning curves of holdout-set estimate of performance
# vs nrounds:
r = range(tree_model, :nrounds, lower=1, upper=600)
curve = learning_curve!(tree, nested_range=(nrounds=r,),
                        measure=rms,
                        resampling=Holdout(fraction_train=0.7),
                        n=4)

using Plots
plot(curve.parameter_values,
     curve.measurements,
     xlab="nrounds", label="performance on holdout",
     ylab="rms")

# Or a CV estimate of performance vs nrounds:
r = range(tree_model, :nrounds, lower=1, upper=300)
curve = learning_curve!(tree, nested_range=(nrounds=r,),
                        measure=rms,
                        resampling=CV(nfolds=5))
plot!(curve.parameter_values,
     curve.measurements,
     label="performance via CV")


