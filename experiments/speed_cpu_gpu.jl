using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

params_c = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, nbins=32);

model_c, cache_c = EvoTrees.init_evotree(params_c, X_train, Y_train);

# initialize from cache
params_c = model_c.params
train_nodes = cache_c.train_nodes
splits = cache_c.splits
X_size = size(cache_c.X_bin)

# select random rows and cols
𝑖 = cache_c.𝑖_[sample(params.rng, cache_c.𝑖_, ceil(Int, params_c.rowsample * X_size[1]), replace=false, ordered=true)]
𝑗 = cache_c.𝑗_[sample(params.rng, cache_c.𝑗_, ceil(Int, params_c.colsample * X_size[2]), replace=false, ordered=true)]
# reset gain to -Inf
for feat in cache_c.𝑗_
    splits[feat].gain = -Inf
end

# build a new tree
# 897.800 μs (6 allocations: 736 bytes)
@btime EvoTrees.update_grads!(params_c.loss, params_c.α, cache_c.pred, cache_c.Y, cache_c.δ, cache_c.δ², cache_c.𝑤)
∑δ, ∑δ², ∑𝑤 = sum(cache_c.δ[𝑖]), sum(cache_c.δ²[𝑖]), sum(cache_c.𝑤[𝑖])
gain = EvoTrees.get_gain(params_c.loss, ∑δ, ∑δ², ∑𝑤, params_c.λ)
# assign a root and grow tree
train_nodes[1] = EvoTrees.TrainNode(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
# 69.247 ms (1852 allocations: 38.41 MiB)
@btime tree = grow_tree(cache_c.δ, cache_c.δ², cache_c.𝑤, cache_c.hist_δ, cache_c.hist_δ², cache_c.hist_𝑤, params_c, train_nodes, splits, cache_c.edges, cache_c.X_bin);
push!(model_c.trees, tree)
@btime EvoTrees.predict!(cache_c.pred, tree, cache_c.X)

###########################
# Tree CPU
###########################
δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, edges, X_bin = cache_c.δ, cache_c.δ², cache_c.𝑤, cache_c.hist_δ, cache_c.hist_δ², cache_c.hist_𝑤, cache_c.edges, cache_c.X_bin;

T = Float32
L = 1
active_id = ones(Int, 1)
leaf_count = one(Int)
tree_depth = one(Int)
tree = EvoTrees.Tree(Vector{EvoTrees.TreeNode{L,T,Int,Bool}}())

id = 1
node = train_nodes[id]
# 9.613 ms (81 allocations: 13.55 KiB)
@btime EvoTrees.update_hist!(hist_δ[id], hist_δ²[id], hist_𝑤[id], δ, δ², 𝑤, X_bin, node)
# 601.685 ns (6 allocations: 192 bytes) 8 100 feat ~ 60us
@btime EvoTrees.find_split!(view(hist_δ[id], :, j), view(hist_δ²[id], :, j), view(hist_𝑤[id], :, j), params, node, splits[j], edges[j])

###################################################
# GPU
###################################################
params_g = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=1.0, colsample=1.0, nbins=32);

model_g, cache_g = EvoTrees.init_evotree_gpu(params_g, X_train, Y_train);

params_g = model_g.params;
train_nodes = cache_g.train_nodes;
splits = cache_g.splits;
X_size = size(cache_g.X_bin);

# select random rows and cols
𝑖 = cache_g.𝑖_[sample(params_g.rng, cache_g.𝑖_, ceil(Int, params_g.rowsample * X_size[1]), replace=false, ordered=true)]
𝑗 = cache_g.𝑗_[sample(params_g.rng, cache_g.𝑗_, ceil(Int, params_g.colsample * X_size[2]), replace=false, ordered=true)]
# reset gain to -Inf
for feat in cache_g.𝑗_
    splits[feat].gain = -Inf
end

# build a new tree
# 164.200 μs (33 allocations: 1.66 KiB) - 5-6 X time faster on GPU
@btime CUDA.@sync EvoTrees.update_grads_gpu!(params_g.loss, cache_g.δ, cache_g.δ², cache_g.pred, cache_g.Y, cache_g.𝑤)
# sum Gradients of each of the K parameters and bring to CPU
∑δ, ∑δ², ∑𝑤 = Vector(vec(sum(cache_g.δ[𝑖,:], dims=1))), Vector(vec(sum(cache_g.δ²[𝑖,:], dims=1))), sum(cache_g.𝑤[𝑖])
gain = EvoTrees.get_gain(params_g.loss, ∑δ, ∑δ², ∑𝑤, params_g.λ)
# assign a root and grow tree
train_nodes[1] = EvoTrees.TrainNode_gpu(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
# 60.736 ms (108295 allocations: 47.95 MiB) - only 15% faster than CPU
@btime CUDA.@sync tree = EvoTrees.grow_tree_gpu(cache_g.δ, cache_g.δ², cache_g.𝑤, cache_g.hist_δ, cache_g.hist_δ², cache_g.hist_𝑤, params_g, cache_g.K, train_nodes, splits, cache_g.edges, cache_g.X_bin, cache_g.X_bin_cpu);
push!(model_g.trees, tree);
# 2.736 ms (93 allocations: 13.98 KiB)
@btime CUDA.@sync EvoTrees.predict_gpu!(cache_g.pred_cpu, tree, cache_g.X)
# 1.013 ms (37 allocations: 1.19 KiB)
@btime CUDA.@sync cache_g.pred .= CuArray(cache_g.pred_cpu);


###########################
# Tree GPU
###########################
δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, K, edges, X_bin, X_bin_cpu = cache_g.δ, cache_g.δ², cache_g.𝑤, cache_g.hist_δ, cache_g.hist_δ², cache_g.hist_𝑤, cache_g.K, cache_g.edges, cache_g.X_bin, cache_g.X_bin_cpu;
T = Float32
active_id = ones(Int, 1)
leaf_count = one(Int)
tree_depth = one(Int)
tree = EvoTrees.Tree_gpu(Vector{EvoTrees.TreeNode_gpu{T,Int,Bool}}())

hist_δ_cpu = zeros(T, size(hist_δ[1]));
hist_δ²_cpu = zeros(T, size(hist_δ²[1]));
hist_𝑤_cpu = zeros(T, size(hist_𝑤[1]));

id = 1
node = train_nodes[id];
# 7.003 ms (106 allocations: 3.09 KiB)
@btime CUDA.@sync EvoTrees.update_hist_gpu!(hist_δ[id], hist_δ²[id], hist_𝑤[id], δ, δ², 𝑤, X_bin, CuVector(node.𝑖), CuVector(node.𝑗), K);
#  32.001 μs (2 allocations: 32 bytes) * 3 adds 100us
@btime hist_δ_cpu .= hist_δ[id];
j = 1
# 2.925 μs (78 allocations: 6.72 KiB) * 100 features ~ 300us
@btime EvoTrees.find_split_gpu!(view(hist_δ_cpu, :, :, j), view(hist_δ²_cpu, :, :, j), view(hist_𝑤_cpu, :, j), params, node, splits[j], edges[j])