using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools

using CSV, DataFrames, MLJBase, EvoTrees
using StableRNGs

data = CSV.read(joinpath(@__DIR__, "..", "data", "debug", "pb_data.csv"), DataFrame)
y = categorical(data.target)
X = data[!, Not(:target)]

train, test = MLJBase.train_test_pairs(Holdout(), 1:size(X, 1), X, y)[1]
rng = StableRNG(6)
model = EvoTreeClassifier(nrounds=5, lambda=1e-5, max_depth=7, rng=rng)
Xtrain, ytrain = MLJBase.reformat(model, selectrows(X, train), selectrows(y, train))
MLJBase.fit(model, 1, Xtrain, ytrain);

# EvoTrees params
rng = StableRNG(6)
params_evo = EvoTreeClassifier(;
    T=Float32,
    nrounds=5,
    lambda=0.0,
    gamma=0.0,
    eta=0.1,
    max_depth=7,
    min_weight=1.0,
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    rng
)

using CategoricalArrays
x_train = Xtrain[:matrix]
y_train = CategoricalArrays.levelcode.(ytrain)

mean(y_train)
sum(ytrain .== true) ./ length(y_train)

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train);
