using Revise
using EvoTrees
using MLUtils
using CSV
using DataFrames
using CategoricalArrays
using Arrow
using CUDA
using Base.Iterators: partition
using Base.Threads: nthreads, @threads
using Tables

nrounds = 10
nobs = Int(1e6)
nfeats_num = Int(10)
T = Float32
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $nfeats_num features."
x_train = rand(T, nobs, nfeats_num);
y_train = rand(T, nobs);

dtrain = DataFrame(x_train, :auto);
dtrain[:, :y] = y_train;
dtrain[:, :x_cat_1] = rand(["lvl1", "lvl2", "lvl3"], nobs);
transform!(dtrain, "x_cat_1" => (x -> categorical(x, ordered = false)) => "x_cat_1")
levels(dtrain.x_cat_1)
levelcode.(dtrain.x_cat_1)
isordered(dtrain.x_cat_1)

@info nthread
loss = "linear"
if loss == "linear"
    loss_evo = :linear
    metric_evo = :mae
elseif loss == "logistic"
    loss_evo = :logistic
    metric_evo = :logloss
end

hyper = EvoTreeRegressor(
    T=T,
    loss=loss_evo,
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
    rng = 123,
)

target_name = "y"
eltype.(eachcol(dtrain))
typeof.(eachcol(dtrain))

@time for col in eachcol(dtrain)
    @info typeof(col)
end
@time for name in names(dtrain)
    @info typeof(dtrain[:, name])
end

@time model, cache = EvoTrees.init_evotree_df(hyper, dtrain; target_name);
@time EvoTrees.grow_evotree!(model, cache);