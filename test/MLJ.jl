using StatsBase: sample
using EvoTrees: sigmoid, logit
using MLJBase
using MLJTestInterface

@testset "generic interface tests" begin
    @testset "EvoTreeRegressor, EvoTreeMLE, EvoTreeGaussian" begin
        failures, summary = MLJTestInterface.test(
            [EvoTreeRegressor, EvoTreeMLE, EvoTreeGaussian],
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "EvoTreeCount" begin
        failures, summary = MLJTestInterface.test(
            [EvoTreeCount],
            MLJTestInterface.make_count()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "EvoTreeClassifier" begin
        for data in [
            MLJTestInterface.make_binary(),
            MLJTestInterface.make_multiclass(),
        ]
            failures, summary = MLJTestInterface.test(
                [EvoTreeClassifier],
                data...;
                mod=@__MODULE__,
                verbosity=0, # bump to debug
                throw=false # set to true to debug
            )
            @test isempty(failures)
        end
    end
end

##################################################
### Regression - small data
##################################################
features = rand(1_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = MLJBase.table(X)

# @load EvoTreeRegressor
# linear regression
tree_model = EvoTreeRegressor(max_depth=5, eta=0.05, nrounds=10)
# logloss - logistic regression
tree_model = EvoTreeRegressor(loss=:logloss, max_depth=5, eta=0.05, nrounds=10)
# quantile regression
# tree_model = EvoTreeRegressor(loss=:quantile, alpha=0.75, max_depth=5, eta=0.05, nrounds=10)

mach = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

# predict on train data
pred_train = predict(mach, selectrows(X, train))
mean(abs.(pred_train - selectrows(Y, train)))

# predict on test data
pred_test = predict(mach, selectrows(X, test))
mean(abs.(pred_test - selectrows(Y, test)))

@test MLJBase.iteration_parameter(EvoTreeRegressor) == :nrounds

##################################################
### Regression - GPU
##################################################
# tree_model = EvoTreeRegressor(loss = :logloss, max_depth = 5, eta = 0.05, nrounds = 10, device = "gpu")
# mach = machine(tree_model, X, y)
# train, test = partition(eachindex(y), 0.7, shuffle = true); # 70:30 split
# fit!(mach, rows = train, verbosity = 1)

# mach.model.nrounds += 10
# fit!(mach, rows = train, verbosity = 1)

# # predict on train data
# pred_train = predict(mach, selectrows(X, train))
# mean(abs.(pred_train - selectrows(Y, train)))

# # predict on test data
# pred_test = predict(mach, selectrows(X, test))
# mean(abs.(pred_test - selectrows(Y, test)))

# @test MLJBase.iteration_parameter(EvoTreeRegressor) == :nrounds

##################################################
### classif - categorical target
##################################################
X, y = @load_crabs

tree_model = EvoTreeClassifier(
    max_depth=4,
    eta=0.05,
    lambda=0.0,
    gamma=0.0,
    nrounds=10,
)

# @load EvoTreeRegressor
mach = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1);

mach.model.nrounds += 50
fit!(mach, rows=train, verbosity=1)

pred_train = predict(mach, selectrows(X, train))
pred_train_mode = predict_mode(mach, selectrows(X, train))
sum(pred_train_mode .== y[train]) / length(y[train])

pred_test = predict(mach, selectrows(X, test))
pred_test_mode = predict_mode(mach, selectrows(X, test))
sum(pred_test_mode .== y[test]) / length(y[test])
pred_test_mode = predict_mode(mach, selectrows(X, test))

##################################################
### count
##################################################
features = rand(1_000, 10)
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
    lambda=0.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = MLJBase.table(X)

# typeof(X)
mach = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

pred = predict(mach, selectrows(X, train))
pred_mean = predict_mean(mach, selectrows(X, train))
pred_mode = predict_mode(mach, selectrows(X, train))
# pred_mode = predict_median(mach, selectrows(X,train))

##################################################
### Gaussian - Larger data
##################################################
features = rand(1_000, 10)
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
    nrounds=10,
    lambda=0.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = MLJBase.table(X)

# typeof(X)
mach = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

pred = predict(mach, selectrows(X, train))
pred_mean = predict_mean(mach, selectrows(X, train))
pred_mode = predict_mode(mach, selectrows(X, train))
# pred_mode = predict_median(mach, selectrows(X,train))
mean(abs.(pred_mean - selectrows(Y, train)))

q_20 = quantile.(pred, 0.20)
q_20 = quantile.(pred, 0.80)

report(mach)

##################################################
### LogLoss - Larger data
##################################################
features = rand(1_000, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeMLE(
    loss=:logistic_mle,
    nrounds=10,
    lambda=1.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=32.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = MLJBase.table(X)

# typeof(X)
mach = machine(tree_model, X, Y)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

pred = predict(mach, selectrows(X, train))
pred_mean = predict_mean(mach, selectrows(X, train))
pred_mode = predict_mode(mach, selectrows(X, train))
# pred_mode = predict_median(mach, selectrows(X,train))
mean(abs.(pred_mean - selectrows(Y, train)))

q_20 = quantile.(pred, 0.20)
q_20 = quantile.(pred, 0.80)

report(mach)

############################
# Added in response to #92 #
############################

# tests that `update` handles data correctly in the case of a cold
# restatrt:

X = MLJBase.table(rand(5, 2))
y = rand(5)
model = EvoTreeRegressor()
data = MLJBase.reformat(model, X, y);
f, c, r = MLJBase.fit(model, 2, data...);
model.lambda = 0.1
MLJBase.update(model, 2, f, c, data...);

@testset "update cold restart" begin

    seed = 123
    X = MLJBase.table(rand(200, 3))
    y = rand(200)

    # Changing any hyperparameter other than `nrounds` invalidates the trees already
    # grown, so `update` must discard them and refit rather than continue boosting.
    model = EvoTreeRegressor(; nrounds=10, eta=0.05, max_depth=3, seed)
    data = MLJBase.reformat(model, X, y)
    f, c, _ = MLJBase.fit(model, 0, data...)
    p_stale = MLJBase.predict(model, f, X)

    model2 = EvoTreeRegressor(; nrounds=10, eta=0.9, max_depth=6, seed)
    f2, c2, _ = MLJBase.update(model2, 0, f, c, data...)
    p_update = MLJBase.predict(model2, f2, X)

    f3, _, _ = MLJBase.fit(model2, 0, data...)
    p_fresh = MLJBase.predict(model2, f3, X)

    @test !all(p_update .≈ p_stale)
    @test all(p_update .≈ p_fresh)

    # Reducing `nrounds` below the number of trees already grown also requires a refit.
    model3 = EvoTreeRegressor(; nrounds=4, eta=0.05, max_depth=3, seed)
    f4, _, _ = MLJBase.update(model3, 0, f, c, data...)
    @test f4.info[:nrounds] == 4

    # Increasing `nrounds` alone is the one case that may continue from the existing
    # ensemble, and doing so must agree with a fit from scratch.
    model4 = EvoTreeRegressor(; nrounds=25, eta=0.05, max_depth=3, seed)
    f5, _, _ = MLJBase.update(model4, 0, f, c, data...)
    f6, _, _ = MLJBase.fit(model4, 0, data...)
    @test f5.info[:nrounds] == 25
    @test all(
        MLJBase.predict(model4, f5, X) .≈
        MLJBase.predict(model4, f6, X),
    )
end


############################
# Feature Importances
############################

# Test feature importances are defined
for model ∈ [
    EvoTreeClassifier(),
    EvoTreeCount(),
    EvoTreeRegressor(),
    EvoTreeMLE(),
    EvoTreeGaussian(),
]

    @test reports_feature_importances(model) == true
end


# Test that feature importances work for Classifier
X, y = MLJBase.make_blobs(100, 3)
model = EvoTreeClassifier()
m = machine(model, X, y)
fit!(m)

rpt = MLJBase.report(m)
fi = MLJBase.feature_importances(model, m.fitresult, rpt)
@test size(fi, 1) == 3

X, y = MLJBase.make_regression(100, 3)
model = EvoTreeRegressor()
m = machine(model, X, y)
fit!(m)

rpt = MLJBase.report(m)
fi = MLJBase.feature_importances(model, m.fitresult, rpt)
@test size(fi, 1) == 3



##################################################
### Test with weights
##################################################
features = rand(1_000, 10)
X = features
Y = rand(size(X, 1))
W = rand(size(X, 1)) .+ 0.1
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]
w_train, w_eval = W[𝑖_train], W[𝑖_eval]

# @load EvoTreeRegressor
tree_model = EvoTreeRegressor(
    loss=:logloss,
    nrounds=10,
    lambda=1.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=32.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=32,
)

X = MLJBase.table(X)

# typeof(X)
mach = machine(tree_model, X, Y, W)
train, test = partition(eachindex(Y), 0.8, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1, force=true)

mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

report(mach)

@testset "MLJ - rowtables - EvoTreeRegressor" begin
    X, y = make_regression(1000, 5)
    X = Tables.rowtable(X)
    booster = EvoTreeRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    predict(mach, X)
end

@testset "MLJ - matrix - EvoTreeRegressor" begin
    X, y = make_regression(1000, 5)
    X = Tables.matrix(X)
    booster = EvoTreeRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    predict(mach, X)
end

##################################################
### issue #267: ordered target
##################################################
@testset "MLJ - supported ordered factor predictions" begin
    X = (; x=rand(10))
    y = coerce(rand("ab", 10), OrderedFactor)
    model = EvoTreeClassifier()
    mach = machine(model, X, y) |> fit!
    yhat = predict(mach, X)
    @assert isordered(yhat)
end
