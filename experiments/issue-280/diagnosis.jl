using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using DataFrames
using BenchmarkTools
using Random: seed!
using CUDA

nobs = Int(1e6)
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
# laptop depth 11:  3.887943 seconds (4.06 M allocations: 1.500 GiB, 4.65% gc time)
# desktop 100K - depth-11: 4.538235 seconds (5.77 M allocations: 1.627 GiB, 3.49% gc time)
# desktop 1M - depth-6: 4.993774 seconds (865.10 k allocations: 306.904 MiB)
# desktop 1M - depth-11: 15.564866 seconds (8.14 M allocations: 1.557 GiB, 1.27% gc time)
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
