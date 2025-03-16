using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using DataFrames
using BenchmarkTools
using Random: seed!
# using CUDA

nobs = Int(1e5)
num_feat = Int(100)
nrounds = 200
T = Float64
nthread = Base.Threads.nthreads()
seed!(123)
x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

@info "EvoTrees"
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
target_name = :y

config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=200,
    eta=0.1,
    L2=1,
    max_depth=11,
    rowsample=0.5,
    colsample=0.5,
    min_weight=1,
    nbins=64,
    device=:cpu,
    tree_type=:binary
)

@info "fit"
# laptop 100k - depth-11: 5.146400 seconds (3.60 M allocations: 1.492 GiB, 4.15% gc time)
# laptop 1M - depth-11: 22.864825 seconds (5.09 M allocations: 1.794 GiB, 2.36% gc time)
# desktop 1M - depth-6: 5.438947 seconds (726.77 k allocations: 293.581 MiB, 1.07% gc time)
# desktop 1M - depth-11: 29.500740 seconds (7.74 M allocations: 2.008 GiB, 1.73% gc time)
@time m = EvoTrees.fit(config, dtrain; target_name)
@profview EvoTrees.fit(config, dtrain; target_name)

@info "init"
@time m, cache = EvoTrees.init(config, dtrain, EvoTrees.CPU; target_name);

@time EvoTrees.grow_evotree!(m, cache, config)
@btime EvoTrees.grow_evotree!(m, cache, config)
@code_warntype EvoTrees.grow_evotree!(m, cache, config)

function grow_profile(n)
    for i in 1:n
        EvoTrees.grow_evotree!(m, cache, config)
    end
end

using Profile
Profile.init(delay=0.0001)
Profile.init()
@profview EvoTrees.grow_evotree!(m, cache, config)
@time grow_profile(200)
@profview grow_profile(200)

# Profile.init()

# @profile EvoTrees.grow_evotree!(m, cache, config)
@profview EvoTrees.grow_evotree!(m, cache, config)

using ProfileView
ProfileView.@profview EvoTrees.grow_evotree!(m, cache, config)
