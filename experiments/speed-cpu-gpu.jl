using Revise
using Statistics
using StatsBase: sample, sample!
using EvoTrees
using BenchmarkTools
using CUDA

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]


###########################
# Tree CPU
###########################
params_c = EvoTreeRegressor(
    T = Float32,
    loss = :linear,
    nrounds = 100,
    lambda = 0.1,
    gamma = 0.0,
    eta = 0.1,
    max_depth = 6,
    min_weight = 1.0,
    rowsample = 0.5,
    colsample = 0.5,
    nbins = 64,
);

# params_c = EvoTrees.EvoTreeLogistic(T=Float32,
#     loss=:linear,
#     nrounds=100,
#     lambda=1.0, gamma=0.0, eta=0.1,
#     max_depth=6, min_weight=1.0,
#     rowsample=0.5, colsample=0.5, nbins=64);

# params_c = EvoTreeGaussian(T=Float32,
#     loss=:gaussian, metric=:none,
#     nrounds=100,
#     λ=1.0, γ=0.1, η=0.1,
#     max_depth=6, min_weight=1.0,
#     rowsample=0.5, colsample=0.5, nbins=64);

model_c, cache_c = EvoTrees.init_evotree(params_c; x_train, y_train);

# initialize from cache
X_size = size(cache_c.x_bin)

𝑗 = cache_c.𝑗

L = EvoTrees.Linear
K = 1
T = Float32
# build a new tree
# 100.736 μs (73 allocations: 6.33 KiB)
@time EvoTrees.update_grads!(cache_c.δ𝑤, cache_c.pred, cache_c.y, params_c)
# @btime EvoTrees.update_grads!($cache_c.δ𝑤, $cache_c.pred, $cache_c.y, $params_c)

# 62.530 ms (7229 allocations: 17.43 MiB)
# 1.25e5: 9.187 ms (7358 allocations: 2.46 MiB)
tree = EvoTrees.Tree{L,K,T}(params_c.max_depth)
@time EvoTrees.grow_tree!(
    tree,
    cache_c.nodes,
    params_c,
    cache_c.δ𝑤,
    cache_c.edges,
    cache_c.𝑗,
    cache_c.left,
    cache_c.left,
    cache_c.right,
    cache_c.x_bin,
    cache_c.monotone_constraints,
)
@code_warntype EvoTrees.grow_tree!(
    tree,
    cache_c.nodes,
    params_c,
    cache_c.δ𝑤,
    cache_c.edges,
    cache_c.𝑗,
    cache_c.left,
    cache_c.left,
    cache_c.right,
    cache_c.x_bin,
    cache_c.monotone_constraints,
)
@btime EvoTrees.grow_tree!(
    $EvoTrees.Tree(params_c.max_depth,),
    $cache_c.nodes,
    $params_c,
    $cache_c.δ𝑤,
    $cache_c.edges,
    $cache_c.𝑗,
    $cache_c.left,
    $cache_c.left,
    $cache_c.right,
    $cache_c.X_bin,
    $cache_c.K,
)
# push!(model_c.trees, tree)
# 1.883 ms (83 allocations: 13.77 KiB)
@btime EvoTrees.predict!(model_c.params.loss, cache_c.pred_cpu, tree, cache_c.X, model_c.K)

δ𝑤, edges, x_bin, nodes, nidx = cache_c.δ𝑤,
cache_c.edges,
cache_c.x_bin,
cache_c.nodes,
cache_c.nidx;

# 7.914 ms (49 allocations: 5.36 KiB)
@time EvoTrees.update_hist!(L, nodes, nidx, δ𝑤, x_bin, 𝑗, K)
@btime EvoTrees.update_hist!($L, $nodes, $nidx, $δ𝑤, $x_bin, $𝑗, $K)
@btime EvoTrees.update_hist!($nodes[1].h, $δ𝑤, $X_bin, $nodes[1].𝑖, $𝑗)
@code_warntype EvoTrees.update_hist!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)

j = 1
# 8.399 μs (80 allocations: 13.42 KiB)
n = 1
nodes[1].∑ .= vec(sum(δ𝑤[:, 𝑖], dims = 2))
EvoTrees.update_gains!(params_c.loss, nodes[n], 𝑗, params_c, K)
nodes[1].gains
# findmax(nodes[1].gains) #1.25e5: 36.500 μs (81 allocations: 8.22 KiB)
@btime EvoTrees.update_gains!($params_c.loss, $nodes[n], $𝑗, $params_c, $K)
@code_warntype EvoTrees.update_gains!(params_c.loss, nodes[n], 𝑗, params_c, K)

#1.25e5: 14.100 μs (1 allocation: 32 bytes)
best = findmax(nodes[n].gains)
@btime best = findmax(nodes[n].gains)
@btime best = findmax(view(nodes[n].gains, :, 𝑗))

tree.cond_bin[n] = best[2][1]
tree.feat[n] = best[2][2]

Int.(tree.cond_bin[n])
# tree.cond_bin[n] = 32


###################################################
# GPU
###################################################
params_g = EvoTreeRegressor(
    T = Float32,
    loss = :linear,
    nrounds = 100,
    lambda = 1.0,
    gamma = 0.1,
    eta = 0.1,
    max_depth = 6,
    min_weight = 1.0,
    rowsample = 0.5,
    colsample = 0.5,
    nbins = 64,
);

model_g, cache_g = EvoTrees.init_evotree_gpu(params_g; x_train, y_train);

x_size = size(cache_g.x_bin);

# select random rows and cols
𝑖c = cache_g.𝑖_[sample(
    params_g.rng,
    cache_g.𝑖_,
    ceil(Int, params_g.rowsample * x_size[1]),
    replace = false,
    ordered = true,
)]
𝑖 = CuVector(𝑖c)
𝑗c = cache_g.𝑗_[sample(
    params_g.rng,
    cache_g.𝑗_,
    ceil(Int, params_g.colsample * x_size[2]),
    replace = false,
    ordered = true,
)]
𝑗 = CuVector(𝑗c)
cache_g.nodes[1].𝑖 = 𝑖
cache_g.𝑗 .= 𝑗c

L = EvoTrees.Linear
K = 1
T = Float32
# build a new tree
# 144.600 μs (23 allocations: 896 bytes) - 5-6 X time faster on GPU
@time CUDA.@sync EvoTrees.update_grads_gpu!(cache_g.δ𝑤, cache_g.pred, cache_g.y, params_g)
# @btime CUDA.@sync EvoTrees.update_grads_gpu!($params_g.loss, $cache_g.δ𝑤, $cache_g.pred_gpu, $cache_g.Y_gpu)
# sum Gradients of each of the K parameters and bring to CPU

# 33.447 ms (6813 allocations: 307.27 KiB)
tree = EvoTrees.TreeGPU{L,K,T}(params_g.max_depth)
sum(cache_g.δ𝑤[:, cache_g.nodes[1].𝑖], dims = 2)
CUDA.@time EvoTrees.grow_tree_gpu!(
    tree,
    cache_g.nodes,
    params_g,
    cache_g.δ𝑤,
    cache_g.edges,
    CuVector(cache_g.𝑗),
    cache_g.out,
    cache_g.left,
    cache_g.right,
    cache_g.x_bin,
    cache_g.monotone_constraints,
)
@btime EvoTrees.grow_tree_gpu!(
    EvoTrees.TreeGPU{L,K,T}(params_g.max_depth),
    cache_g.nodes,
    params_g,
    $cache_g.δ𝑤,
    $cache_g.edges,
    $𝑗,
    $cache_g.out,
    $cache_g.left,
    $cache_g.right,
    $cache_g.x_bin,
    $cache_g.monotone_constraints,
);
@code_warntype EvoTrees.grow_tree_gpu!(
    EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ),
    params_g,
    cache_g.δ,
    cache_g.hist,
    cache_g.histL,
    cache_g.histR,
    cache_g.gains,
    cache_g.edges,
    𝑖,
    𝑗,
    𝑛,
    cache_g.x_bin,
);

push!(model_g.trees, tree);
# 2.736 ms (93 allocations: 13.98 KiB)
@time CUDA.@sync EvoTrees.predict!(cache_g.pred_gpu, tree, cache_g.X_bin)
@btime CUDA.@sync EvoTrees.predict!($cache_g.pred_gpu, $tree, $cache_g.X_bin)

###########################
# Tree GPU
###########################
δ𝑤, edges, x_bin, nodes, out, left, right = cache_g.δ𝑤,
cache_g.edges,
cache_g.x_bin,
cache_g.nodes,
cache_g.out,
cache_g.left,
cache_g.right;

# 2.571 ms (1408 allocations: 22.11 KiB)
# 𝑗2 = CuArray(sample(UInt32.(1:100), 50, replace=false, ordered=true))
@time EvoTrees.update_hist_gpu!(nodes[1].h, δ𝑤, x_bin, 𝑖, 𝑗)
@btime EvoTrees.update_hist_gpu!($nodes[1].h, $δ𝑤, $x_bin, $𝑖, $𝑗)
# @code_warntype EvoTrees.update_hist_gpu!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)


# 72.100 μs (186 allocations: 6.00 KiB)
n = 1
nodes[1].∑ .= vec(sum(δ𝑤[:, 𝑖], dims = 2))
CUDA.@time EvoTrees.update_gains_gpu!(params_g.loss, nodes[n], 𝑗, params_g, K)
@btime EvoTrees.update_gains_gpu!($params_g.loss, $nodes[n], $𝑗, $params_g, $K)

tree = EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ)
best = findmax(nodes[n].gains)
if best[2][1] != params_g.nbins && best[1] > nodes[n].gain + params_g.γ
    tree.gain[n] = best[1]
    tree.cond_bin[n] = best[2][1]
    tree.feat[n] = best[2][2]
    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
end

tree.split[n] = tree.cond_bin[n] != 0
tree.feat[n]
Int(tree.cond_bin[n])

# 673.900 μs (600 allocations: 29.39 KiB)
offset = 0
_left, _right = EvoTrees.split_set_threads_gpu!(
    out,
    left,
    right,
    𝑖,
    X_bin,
    tree.feat[n],
    tree.cond_bin[n],
    offset,
)
@time EvoTrees.split_set_threads_gpu!(
    out,
    left,
    right,
    𝑖,
    X_bin,
    tree.feat[n],
    tree.cond_bin[n],
    offset,
)
@btime EvoTrees.split_set_threads_gpu!(
    $out,
    $left,
    $right,
    $𝑖,
    $X_bin,
    $tree.feat[n],
    $tree.cond_bin[n],
    $offset,
)
