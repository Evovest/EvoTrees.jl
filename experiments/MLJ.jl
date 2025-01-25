using Tables
using DataFrames
using MLJBase
using StatsBase: sample, mean, quantile
using Statistics
using CategoricalArrays
using Distributions
using EvoTrees
using EvoTrees: logit, sigmoid
using CUDA

##################################################
### Regression - small data
##################################################
features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = Tables.table(X)
X = DataFrame(X)

# @load EvoTreeRegressor
# linear regression
learner = EvoTreeRegressor(; loss=:mse, max_depth=5, eta=0.05, nrounds=5, rowsample=0.5, device=:cpu)
tree = machine(learner, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(tree, rows=train, verbosity=1);

tree.model.nrounds += 5
fit!(tree, rows=train, verbosity=1);
tree.fitresult.trees

# predict on train data
pred_train = predict(tree, selectrows(X, train))
println(mean(abs.(pred_train - selectrows(Y, train))))

# predict on test data
pred_test = predict(tree, selectrows(X, test))
println(mean(abs.(pred_test - selectrows(Y, test))))


##################################################
### classif
##################################################
X, y_train = @load_crabs
x_train = matrix(X)

using CUDA
CUDA.allowscalar(false)

# define hyperparameters
config = EvoTreeClassifier(
    max_depth=4,
    eta=0.05,
    lambda=0.0,
    gamma=0.0,
    nbins=32,
    nrounds=200,
)
model = fit_evotree(config; x_train, y_train);
model = fit_evotree(config; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=:mlogloss, print_every_n=10, early_stopping_rounds=25);

pred = model(x_train)
pred_cat = pred .> 0.5
sum((y_train .== "B") .== pred_cat[:, 1]) / length(y_train)

# @load EvoTreeRegressor
mach = machine(config, X, y_train)
train, test = partition(eachindex(y_train), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)
rpt = report(mach)
MLJBase.feature_importances(config, mach.fitresult, rpt)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)
rpt = report(mach)
MLJBase.feature_importances(config, mach.fitresult, rpt)

pred_train = EvoTrees.predict(mach, selectrows(X, train))
pred_train_mode = predict_mode(mach, selectrows(X, train))
println(cross_entropy(pred_train, selectrows(y_train, train)) |> mean)
println(sum(pred_train_mode .== y_train[train]) / length(train))

pred_test = EvoTrees.predict(mach, selectrows(X, test))
pred_test_mode = predict_mode(mach, selectrows(X, test))
println(cross_entropy(pred_test, selectrows(y_train, test)) |> mean)
println(sum(pred_test_mode .== y_train[test]) / length(test))
pred_test_mode = predict_mode(mach, selectrows(X, test))

# using LossFunctions, Plots
# evo_model = EvoTreeClassifier(max_depth=6, η=0.05, λ=1.0, γ=0.0, nrounds=10, nbins=64)
# evo = machine(evo_model, X, y)
# r = range(evo_model, :nrounds, lower=1, upper=500)
# @time curve = learning_curve!(evo, range=r, resolution=10, measure=HingeLoss())
# plot(curve.parameter_values, curve.measurements)

##################################################
### regression - Larger data
##################################################
features = rand(1_000_000, 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeRegressor(
    loss=:linear,
    metric=:mae,
    nrounds=10,
    λ=0.0,
    γ=0.0,
    η=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = Tables.table(X);
X = Tables.rowtable(X);
# X = Tables.columntable(X)
# X_matrix = MLJBase.matrix(X)

# typeof(X)
@time tree = machine(tree_model, X, Y);
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time fit!(tree, rows=train, verbosity=1, force=true)

using LossFunctions
using MLJ
r = range(tree_model, :nrounds, lower=1, upper=100)
m = rms
@time curve = learning_curve!(evo, range=r, resolution=100, measure=m)

tree.model.nrounds += 1
@time update(tree.model, 0, tree.fitresult, tree.cache, X, Y);

tree.model.nrounds += 1
@time fit!(tree, rows=train, verbosity=1)
# @time MLJBase.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred_train = predict(tree, selectrows(X, train))
mean(abs.(pred_train - selectrows(Y, train)))

##################################################
### count - Larger data
##################################################
features = rand(100_000, 100)
# features = rand(100, 10)
X = features
Y = rand(UInt8, size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeCount(
    loss=:poisson,
    metric=:poisson,
    nrounds=10,
    λ=0.0,
    γ=0.0,
    η=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = Tables.table(X)
X = Tables.rowtable(X)
X = Tables.columntable(X)
X_matrix = MLJBase.matrix(X)

# typeof(X)
@time tree = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time fit!(tree, rows=train, verbosity=1, force=true)

tree.model.nrounds += 10
@time MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, Y)

tree.model.nrounds += 10
@time fit!(tree, rows=train, verbosity=1)
# @time MLJBase.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred = predict(tree, selectrows(X, train))
pred_mean = predict_mean(tree, selectrows(X, train))
pred_mode = predict_mode(tree, selectrows(X, train))

##################################################
### Gaussian - Larger data
##################################################
features = rand(100_000, 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeGaussian(
    loss=:gaussian,
    metric=:gaussian,
    nrounds=10,
    λ=0.0,
    γ=0.0,
    η=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = Tables.table(X)
X_matrix = MLJBase.matrix(X)

# typeof(X)
@time tree = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time fit!(tree, rows=train, verbosity=1, force=true)

tree.model.nrounds += 10
@time MLJBase.update(tree.model, 0, tree.fitresult, tree.cache, X, Y)

tree.model.nrounds += 10
@time fit!(tree, rows=train, verbosity=1)
# @time MLJBase.fit!(tree, rows=train, verbosity=1)

# yhat = MLJBase.predict(tree.model, tree.fitresult, MLJ.selectrows(X,test))
pred = predict(tree, selectrows(X, train))
pred_mean = predict_mean(tree, selectrows(X, train))
pred_mode = predict_mode(tree, selectrows(X, train))
mean(abs.(pred_mean - selectrows(Y, train)))

q_20 = quantile.(pred, 0.20)
q_20 = quantile.(pred, 0.80)



#########################################
# MLJ2 test
#########################################
using EvoTrees
using MLJModelInterface
using MLJBase
using StatsBase: sample, mean, quantile
using Tables

X = rand(1_000_000, 100);
Y = rand(size(X, 1))

# @load EvoTreeRegressor
tree_model = EvoTreeRegressor(
    loss=:linear,
    metric=:mae,
    nrounds=10,
    λ=0.0,
    γ=0.0,
    η=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = Tables.table(X);
# X = Tables.rowtable(X);
# X = Tables.columntable(X);
# X_matrix = MLJBase.matrix(X);

# typeof(X)
@time tree = machine(tree_model, X, Y);
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
@time fit!(tree, rows=train, verbosity=1, force=false)

tree.model.nrounds += 1
@time fit!(tree, rows=train, verbosity=1)
