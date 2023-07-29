using Statistics
using StatsBase:sample
using Base.Threads:@threads
using BenchmarkTools
using Revise
using EvoTrees
using Profile

nobs = Int(1e6)
num_feat = Int(100)
nrounds = 200
nthread = Base.Threads.nthreads()
x_train = rand(nobs, num_feat)
y_train = rand(size(x_train, 1))

config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=1,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=12,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
    tree_type="binary",
    rng=123
)

################################
# high-level
################################
@time m, cache = EvoTrees.init(config, x_train, y_train);
@time EvoTrees.grow_evotree!(m, cache, config)

Profile.clear()
# Profile.init()
Profile.init(n = 10^7, delay = 0.001)
# @profile m, cache = EvoTrees.init(config, x_train, y_train);
@profile EvoTrees.grow_evotree!(m, cache, config)
Profile.print()
@profview EvoTrees.grow_evotree!(m, cache, config)

################################
# mid-level
################################
@time m, cache = EvoTrees.init(config, x_train, y_train);
@time EvoTrees.grow_evotree!(m, cache, config)
# compute gradients
@time m, cache = EvoTrees.init(config, x_train, y_train);
@time EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, config)
# subsample rows
@time cache.nodes[1].is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, config.rowsample, config.rng)
# subsample cols
EvoTrees.sample!(config.rng, cache.js_, cache.js, replace=false, ordered=true)
L = EvoTrees._get_struct_loss(m)
# instantiate a tree then grow it
tree = EvoTrees.Tree{L,1}(config.max_depth)
grow! = config.tree_type == "oblivious" ? EvoTrees.grow_otree! : EvoTrees.grow_tree!
@time EvoTrees.grow_tree!(
    tree,
    cache.nodes,
    config,
    cache.∇,
    cache.edges,
    cache.js,
    cache.out,
    cache.left,
    cache.right,
    cache.x_bin,
    cache.feattypes,
    cache.monotone_constraints
)

using ProfileView
ProfileView.@profview EvoTrees.grow_tree!(
    tree,
    cache.nodes,
    config,
    cache.∇,
    cache.edges,
    cache.js,
    cache.out,
    cache.left,
    cache.right,
    cache.x_bin,
    cache.feattypes,
    cache.monotone_constraints
)

################################
# end mid-level
################################


@time m_evo = grow_tree!(params_evo; x_train, y_train, device, print_every_n=100);

@info "train - no eval"
@time m_evo = fit_evotree(params_evo; x_train, y_train, device, print_every_n=100);


offset = 0
feat = 15
cond_bin = 32
@time l, r = split_set_threads!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset, 2^14);
@btime split_set_threads!($out, $left, $right, $𝑖, $X_bin, $feat, $cond_bin, $offset, 2^14);
@code_warntype split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)

offset = 0
feat = 15
cond_bin = 32
lid1, rid1 = split_set_threads!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset)
offset = 0
feat = 14
cond_bin = 12
lid2, rid2 = split_set_threads!(out, left, right, lid1, X_bin, feat, cond_bin, offset)
offset = + length(lid1)
feat = 14
cond_bin = 12
lid3, rid3 = split_set_threads!(out, left, right, rid1, X_bin, feat, cond_bin, offset)

lid1_ = deepcopy(lid1)



