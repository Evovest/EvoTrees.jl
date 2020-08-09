using Statistics
using StatsBase: sample
# using XGBoost
using Revise
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(Int(2.25e6), 100)
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

config = EvoTrees.EvoTreeRegressor3(T=Float32,
        loss=:linear, metric=:none,
        nrounds=100, α = 0.5,
        λ = 0.0, γ=0.0, η=0.05,
        max_depth = 6, min_weight = 1.0,
        rowsample=0.5, colsample=0.5, nbins=32)


# for 1.25e5 init_evotree: 2.009 s 0.322925 seconds (2.53 k allocations: 167.345 MiB)
# for 1.25e5 no eval iter 100: 2.009 s (628514 allocations: 720.62 MiB)
# for 1.25e6 no eval iter 10: 6.200 s (44330 allocations: 2.19 GiB)
# for 1.25e6 no eval iter 100: 19.481940 seconds (635.33 k allocations: 6.679 GiB, 3.11% gc time)
# for 1.25e6 mse with eval data: 6.321 s (45077 allocations: 2.19 GiB)
@time model, cache = init_evotree(config, X_train, Y_train);
@time grow_evotree!(model, cache);
@time model = fit_evotree(config, X_train, Y_train);
@btime model = fit_evotree(config, X_train, Y_train);
@time pred_train = EvoTrees.predict(model, X_train)

@time model = fit_evotree(config, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=9999, early_stopping_rounds=9999);
@btime model = fit_evotree(config, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n=9999, early_stopping_rounds=9999);

@time model = fit_evotree(config, X_train, Y_train, early_stopping_rounds=10);
@time model = fit_evotree(config, X_train, Y_train, print_every_n=2);

# @time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 5);
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval);
@time pred_train = predict(model, X_train)


#############################
# agaricus
#############################
function readlibsvm(fname::String, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, parse(Int, itm[1]) + 1] = parse(Int, itm[2])
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end

# we use auxiliary function to read LIBSVM format into julia Matrix
train_X, train_Y = readlibsvm("data/agaricus.txt.train", (6513, 126))
test_X, test_Y   = readlibsvm("data/agaricus.txt.test", (1611, 126))

#-------------Basic Training using XGBoost-----------------
# note: xgboost naturally handles sparse input
# use sparse matrix when your feature is sparse(e.g. when you using one-hot encoding vector)
# model parameters can be set as parameters for ```xgboost``` function, or use a Vector{String} / Dict()
num_round = 100
# you can directly pass Julia's matrix or sparse matrix as data,
# by calling xgboost(data, num_round, label=label, training-parameters)
metrics = ["logloss"]
@time bst = xgboost(train_X, num_round, label = train_Y, eta = 0.1, max_depth = 3, metrics = metrics, silent=0, objective = "binary:logistic")
features_xgb = XGBoost.importance(bst)

params1 = EvoTreeRegressor(
    loss=:logistic, metric=:logloss,
    nrounds=100,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 4, min_weight = 1.0,
    rowsample=1.0, colsample=1.0, nbins=250)

@time model = fit_evotree(params1, train_X, train_Y, print_every_n=20);
@time model = fit_evotree(params1, X_train, Y_train, X_eval=test_X, Y_eval=test_Y, print_every_n=20);
@time pred_train = EvoTrees.predict(model, X_train)
features_evo = importance(model, 1:size(X_train,2))
sort(collect(values(features_evo)))
