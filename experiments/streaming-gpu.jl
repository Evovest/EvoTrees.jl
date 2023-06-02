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
using BenchmarkTools
# using StatsBase
# x1 = rand(Bool, 10)
# nbins = 2
# edges = sort(unique(quantile(skipmissing(x1), (1:nbins-1) / nbins)))
# searchsortedfirst(edges, edges[1])
# searchsortedfirst(edges, 1.0)
# searchsortedfirst(edges, edges[9] + 0.01)

nrounds = 200
nobs = Int(1e6)
nfeats_num = Int(100)
T = Float32
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $nfeats_num features."
x_train = rand(T, nobs, nfeats_num);
y_train = rand(T, nobs);

dtrain = DataFrame(x_train, :auto);
dtrain[:, :y] = y_train;

dtrain[:, :x_cat_1] = rand(["lvl1", "lvl2", "lvl3"], nobs);
transform!(dtrain, "x_cat_1" => (x -> categorical(x, ordered=false)) => "x_cat_1")

# levels(dtrain.x_cat_1)
# levelcode.(dtrain.x_cat_1)
# isordered(dtrain.x_cat_1)
# eltype.(eachcol(dtrain))
# typeof.(eachcol(dtrain))
# @time for col in eachcol(dtrain)
#     @info typeof(col)
# end
# @time for name in names(dtrain)
#     @info typeof(dtrain[:, name])
# end

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
    rng=123,
)

target_name = "y"
CUDA.allowscalar(false)
# @time model, cache = EvoTrees.init_evotree_df(hyper, dtrain; target_name, fnames_cat = ["x_cat_1"]);
@time model, cache = EvoTrees.init_evotree_gpu(hyper, dtrain; target_name);
cache.edges[11]
cache.featbins
cache.feattypes
cache.nodes[1].gains[1]
model.trees[1]

@time EvoTrees.grow_evotree!(model, cache, hyper);

for i in 1:10
    @time EvoTrees.grow_evotree!(model, cache, hyper)
end

@btime EvoTrees.grow_evotree!(model, cache, hyper);

@time m = EvoTrees.fit_evotree_df(hyper; dtrain, target_name, verbosity=false);
@btime EvoTrees.fit_evotree_df(hyper; dtrain, target_name, verbosity=false);

@time m = EvoTrees.fit_evotree_df(hyper; dtrain, deval=dtrain, target_name, metric=metric_evo, print_every_n=100, verbosity=false);
@btime m = EvoTrees.fit_evotree_df(hyper; dtrain, deval=dtrain, target_name, metric=metric_evo, print_every_n=100, verbosity=false);

@time pred = m(dtrain);
@btime m($dtrain);
