using StatsBase: sample
using EvoTrees: sigmoid, logit
using MLJBase

##################################################
### Regression - small data
##################################################
features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = MLJBase.table(X)

# @load EvoTreeRegressor
# linear regression
tree_model = EvoTreeRegressor(max_depth=5, η=0.05, nrounds=10)
# logistic regression
tree_model = EvoTreeRegressor(loss=:logistic, max_depth=5, η=0.05, nrounds=10)
# quantile regression
# tree_model = EvoTreeRegressor(loss=:quantile, α=0.75, max_depth=5, η=0.05, nrounds=10)

mach = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

# predict on train data
pred_train = predict(mach, selectrows(X,train))
mean(abs.(pred_train - selectrows(Y,train)))

# predict on test data
pred_test = predict(mach, selectrows(X,test))
mean(abs.(pred_test - selectrows(Y,test)))


##################################################
### classif - categorical target
##################################################
X, y = @load_crabs

tree_model = EvoTreeClassifier(max_depth=4, η=0.05, λ=0.0, γ=0.0, nrounds=10)

# @load EvoTreeRegressor
mach = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)

mach.model.nrounds += 50
fit!(mach, rows=train, verbosity=1)

pred_train = predict(mach, selectrows(X,train))
pred_train_mode = predict_mode(mach, selectrows(X,train))
cross_entropy(pred_train, selectrows(y, train)) |> mean
sum(pred_train_mode .== y[train]) / length(y[train])

pred_test = predict(mach, selectrows(X,test))
pred_test_mode = predict_mode(mach, selectrows(X,test))
cross_entropy(pred_test, selectrows(y, test)) |> mean
sum(pred_test_mode .== y[test]) / length(y[test])
pred_test_mode = predict_mode(mach, selectrows(X,test))

##################################################
### count
##################################################
features = rand(10_000, 10)
# features = rand(100, 10)
X = features
Y = rand(UInt8, size(X, 1))
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeCount(
    loss=:poisson, metric=:poisson,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

X = MLJBase.table(X)
X = MLJBase.matrix(X)

# typeof(X)
mach = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

pred = predict(mach, selectrows(X,train))
pred_mean = predict_mean(mach, selectrows(X,train))
pred_mode = predict_mode(mach, selectrows(X,train))
# pred_mode = predict_median(mach, selectrows(X,train))

##################################################
### Gaussian - Larger data
##################################################
features = rand(10_000, 10)
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
tree_model = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, nbins=32)

X = MLJBase.table(X)

# typeof(X)
mach = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

pred = predict(mach, selectrows(X,train))
pred_mean = predict_mean(mach, selectrows(X,train))
pred_mode = predict_mode(mach, selectrows(X,train))
# pred_mode = predict_median(mach, selectrows(X,train))
mean(abs.(pred_mean - selectrows(Y,train)))

q_20 = quantile.(pred, 0.20)
q_20 = quantile.(pred, 0.80)

report(mach)

############################
# Added in response to #92 #
############################

# tests that `update` handles data correctly in the case of a cold
# restatrt:

X = MLJBase.table(rand(5,2))
y = rand(5)
model = EvoTreeRegressor()
data = MLJBase.reformat(model, X, y);
f, c, r = MLJBase.fit(model, 2, data...);
model.λ = 0.1
MLJBase.update(model, 2, f, c, data...);