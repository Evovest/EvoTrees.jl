using Statistics
using StatsBase:sample
using Revise
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
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1)) + 1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]


###########################
# Tree CPU
###########################
params_c = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=32);

model_c, cache_c = EvoTrees.init_evotree(params_c, X_train, Y_train);

# initialize from cache
params_c = model_c.params
train_nodes = cache_c.train_nodes
splits = cache_c.splits
X_size = size(cache_c.X_bin)

# select random rows and cols
𝑖 = cache_c.𝑖_[sample(params_c.rng, cache_c.𝑖_, ceil(Int, params_c.rowsample * X_size[1]), replace=false, ordered=true)]
𝑗 = cache_c.𝑗_[sample(params_c.rng, cache_c.𝑗_, ceil(Int, params_c.colsample * X_size[2]), replace=false, ordered=true)]
# reset gain to -Inf
for feat in cache_c.𝑗_
    splits[feat].gain = -Inf
end

# build a new tree
# 897.800 μs (6 allocations: 736 bytes)
@time EvoTrees.update_grads!(params_c.loss, params_c.α, cache_c.pred_cpu, cache_c.Y_cpu, cache_c.δ, cache_c.δ², cache_c.𝑤)
∑δ, ∑δ², ∑𝑤 = sum(cache_c.δ[𝑖]), sum(cache_c.δ²[𝑖]), sum(cache_c.𝑤[𝑖])
gain = EvoTrees.get_gain(params_c.loss, ∑δ, ∑δ², ∑𝑤, params_c.λ)
# assign a root and grow tree
train_nodes[1] = EvoTrees.TrainNode(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
# 69.247 ms (1852 allocations: 38.41 MiB)
@btime tree = grow_tree(cache_c.δ, cache_c.δ², cache_c.𝑤, cache_c.hist_δ, cache_c.hist_δ², cache_c.hist_𝑤, params_c, train_nodes, splits, cache_c.edges, cache_c.X_bin);
push!(model_c.trees, tree)
@btime EvoTrees.predict!(cache_c.pred, tree, cache_c.X)

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
@time EvoTrees.update_hist!(hist_δ[id], hist_δ²[id], hist_𝑤[id], δ, δ², 𝑤, X_bin, node)
@btime EvoTrees.update_hist!($hist_δ[id], $hist_δ²[id], $hist_𝑤[id], $δ, $δ², $𝑤, $X_bin, $node)

j = 1
# 601.685 ns (6 allocations: 192 bytes) 8 100 feat ~ 60us
@btime EvoTrees.find_split!(view(hist_δ[id], :, j), view(hist_δ²[id], :, j), view(hist_𝑤[id], :, j), params_c, node, splits[j], edges[j])

for j in node.𝑗
    splits[j].gain = node.gain
    EvoTrees.find_split!(view(hist_δ[id],:,j), view(hist_δ²[id],:,j), view(hist_𝑤[id],:,j), params_c, node, splits[j], edges[j])
end
best_cpu = EvoTrees.get_max_gain(splits)


set = node.𝑖
best = X_bin[3]
@btime EvoTrees.update_set(set, best, X_bin[:,1])
@btime EvoTrees.update_set(node.𝑖, best, view(X_bin, :, 1))

###################################################
# GPU
###################################################
params_g = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=6, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=32);

model_g, cache_g = EvoTrees.init_evotree_gpu(params_g, X_train, Y_train);

params_g = model_g.params;
train_nodes = cache_g.train_nodes;
# splits = cache_g.splits;
X_size = size(cache_g.X_bin);

# select random rows and cols
𝑖 = CuVector(cache_g.𝑖_[sample(params_g.rng, cache_g.𝑖_, ceil(Int, params_g.rowsample * X_size[1]), replace=false, ordered=true)])
𝑗 = CuVector(cache_g.𝑗_[sample(params_g.rng, cache_g.𝑗_, ceil(Int, params_g.colsample * X_size[2]), replace=false, ordered=true)])
# reset gain to -Inf
# splits.gains .= -Inf

# build a new tree
# 144.600 μs (23 allocations: 896 bytes) - 5-6 X time faster on GPU
@time CUDA.@sync EvoTrees.update_grads_gpu!(params_g.loss, cache_g.δ, cache_g.pred_gpu, cache_g.Y)
# sum Gradients of each of the K parameters and bring to CPU
∑δ = Array(vec(sum(cache_g.δ[𝑖,:], dims=1)))
gain = EvoTrees.get_gain_gpu(params_g.loss, ∑δ, params_g.λ)
# assign a root and grow tree
train_nodes[1] = EvoTrees.TrainNodeGPU(UInt32(0), UInt32(1), ∑δ, gain, 𝑖, 𝑗)
# 60.736 ms (108295 allocations: 47.95 MiB) - only 15% faster than CPU

EvoTrees.grow_tree_gpu(cache_g.δ, cache_g.hist, params_g, cache_g.K, train_nodes, cache_g.edges, cache_g.X_bin);
@btime CUDA.@sync tree = EvoTrees.grow_tree_gpu(cache_g.δ, cache_g.δ², cache_g.𝑤, cache_g.hist_δ, cache_g.hist_δ², cache_g.hist_𝑤, params_g, cache_g.K, train_nodes, splits, cache_g.edges, cache_g.X_bin, cache_g.X_bin_cpu);
push!(model_g.trees, tree);
# 2.736 ms (93 allocations: 13.98 KiB)
@btime CUDA.@sync EvoTrees.predict_gpu!(cache_g.pred_cpu, tree, cache_g.X)
# 1.013 ms (37 allocations: 1.19 KiB)
@btime CUDA.@sync cache_g.pred .= CuArray(cache_g.pred_cpu);


###########################
# Tree GPU
###########################
δ, hist, K, edges, X_bin = cache_g.δ, cache_g.hist, cache_g.K, cache_g.edges, cache_g.X_bin;
T = Float32
S = UInt32
active_id = ones(S, 1)
leaf_count = one(S)
tree_depth = one(S)
tree = EvoTrees.TreeGPU(Vector{EvoTrees.TreeNodeGPU{T,S,Bool}}())

id = S(1)
node = train_nodes[id];
# 2.930 ms (24 allocations: 656 bytes)
@time CUDA.@sync EvoTrees.update_hist_gpu!(hist[1], δ, X_bin, node.𝑖, node.𝑗, K);
@btime CUDA.@sync EvoTrees.update_hist_gpu!($hist[1], $δ, $X_bin, $node.𝑖, $node.𝑗, $K, MAX_THREADS=128);

j = 1
# 2.925 μs (78 allocations: 6.72 KiB) * 100 features ~ 300us
EvoTrees.find_split_gpu_v1!(hist[j], params_g, node, splits, edges, node.𝑗, K)
@btime CUDA.@sync EvoTrees.find_split_gpu_v1!(hist[j], edges, params_g)

j = 1
# 347.199 μs (403 allocations: 13.31 KiB)
best_g = EvoTrees.find_split_gpu_v2!(hist[j], edges, params_g);
best_g[1]
best_g

@btime CUDA.@sync EvoTrees.find_split_gpu_v2!(hist[j], edges, params_g);

# 673.900 μs (600 allocations: 29.39 KiB)
left, right = EvoTrees.update_set_gpu(node.𝑖, 16, X_bin[:,1], MAX_THREADS=1024);
@btime CUDA.@sync EvoTrees.update_set_gpu(node.𝑖, 16, X_bin[:,1], MAX_THREADS=1024);