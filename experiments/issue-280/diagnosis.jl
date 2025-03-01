using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using DataFrames
using BenchmarkTools
using Random: seed!

nobs = Int(1e6)
num_feat = Int(10)
nrounds = 1
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
    max_depth=11,
    rowsample=1.0,
    colsample=1.0,
    device=:cpu
)

@info "fit"
# depth 6: ~1.0 sec
# depth 11: ~9.0 sec
@time m = EvoTrees.fit(config, dtrain; target_name)

@info "init"
m, cache = EvoTrees.init(config, dtrain, EvoTrees.CPU; target_name);
@time EvoTrees.grow_evotree!(m, cache, config)
@btime EvoTrees.grow_evotree!(m, cache, config)
@code_warntype EvoTrees.grow_evotree!(m, cache, config)

function grow_profile(n)
    for i in 1:n
        EvoTrees.grow_evotree!(m, cache, config)
    end
end

using Profile
Profile.init(delay=0.000001)
Profile.init()
@profview EvoTrees.grow_evotree!(m, cache, config)
@profview grow_profile(100)

# Profile.init()

# @profile EvoTrees.grow_evotree!(m, cache, config)
@profview EvoTrees.grow_evotree!(m, cache, config)

using ProfileView
ProfileView.@profview EvoTrees.grow_evotree!(m, cache, config)
