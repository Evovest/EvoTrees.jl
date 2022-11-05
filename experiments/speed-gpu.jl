using Statistics
using StatsBase:sample, sample!
using EvoTrees
using BenchmarkTools
using CUDA


# allocations on transfer
function gpu_to_cpu(dst, src)
    copy!(dst, src)
end
function cpu_to_gpu(dst, src)
    copy!(dst, src)
end

x1c = zeros(1000, 1000)
x1g = CUDA.rand(1000, 1000)
@time gpu_to_cpu(x1c, x1g);
CUDA.@time gpu_to_cpu(x1c, x1g);

x1c = rand(1000, 1000)
x1g = CUDA.zeros(1000, 1000)
@time cpu_to_gpu(x1g, x1c);
CUDA.@time cpu_to_gpu(x1g, x1c);

function reshape_gpu(x)
    reshape(x, 4, size(x, 1) ÷ 4, size(x,2))
end
x2 = CUDA.@time reshape_gpu(x1g);
reshape(x1g, 4, 250, 1000)

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

###################################################
# GPU
###################################################
params_g = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=100,
    λ=1.0, γ=0.1, η=0.1,
    max_depth=2, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=64);

model_g, cache_g = EvoTrees.init_evotree_gpu(params_g, X_train, Y_train);

params_g = model_g.params;
X_size = size(cache_g.X_bin);

# select random rows and cols
𝑖c = cache_g.𝑖_[sample(params_g.rng, cache_g.𝑖_, ceil(Int, params_g.rowsample * X_size[1]), replace=false, ordered=true)]
𝑖 = CuVector(𝑖c)
𝑗c = cache_g.𝑗_[sample(params_g.rng, cache_g.𝑗_, ceil(Int, params_g.colsample * X_size[2]), replace=false, ordered=true)]
𝑗 = CuVector(𝑗c)

cache_g.nodes[1].𝑖 = 𝑖
cache_g.𝑗 .= 𝑗
# build a new tree
# 144.600 μs (23 allocations: 896 bytes) - 5-6 X time faster on GPU
@time CUDA.@sync EvoTrees.update_grads_gpu!(params_g.loss, cache_g.δ𝑤, cache_g.pred_gpu, cache_g.Y_gpu)
# @btime CUDA.@sync EvoTrees.update_grads_gpu!($params_g.loss, $cache_g.δ𝑤, $cache_g.pred_gpu, $cache_g.Y_gpu)
# sum Gradients of each of the K parameters and bring to CPU

# 33.447 ms (6813 allocations: 307.27 KiB)
tree = EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ)
CUDA.@time EvoTrees.grow_tree_gpu!(tree, cache_g.nodes, params_g, cache_g.δ𝑤, cache_g.edges, cache_g.𝑗, cache_g.out, cache_g.left, cache_g.right, cache_g.X_bin, cache_g.K)
CUDA.@time EvoTrees.grow_tree_gpu!(tree, cache_g.nodes, params_g, cache_g.δ𝑤, cache_g.edges, cache_g.𝑗, cache_g.out, cache_g.left, cache_g.right, cache_g.X_bin, cache_g.K)

# CUDA.@time EvoTrees.grow_tree_gpu!(tree, params_g, cache_g.δ, cache_g.hist, cache_g.histL, cache_g.histR, cache_g.gains, cache_g.edges, 𝑖, 𝑗, 𝑛, cache_g.X_bin);
# @btime EvoTrees.grow_tree_gpu!(EvoTrees.TreeGPU(UInt32($params_g.max_depth), $model_g.K, $params_g.λ), $params_g, $cache_g.δ, $cache_g.hist, $cache_g.histL, $cache_g.histR, $cache_g.gains, $cache_g.edges, $𝑖, $𝑗, $𝑛, $cache_g.X_bin);
# @code_warntype EvoTrees.grow_tree_gpu!(EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ), params_g, cache_g.δ, cache_g.hist, cache_g.histL, cache_g.histR, cache_g.gains, cache_g.edges, 𝑖, 𝑗, 𝑛, cache_g.X_bin);

# push!(model_g.trees, tree);
# # 2.736 ms (93 allocations: 13.98 KiB)
# @time CUDA.@sync EvoTrees.predict_gpu!(cache_g.pred_gpu, tree, cache_g.X_bin)
# @btime CUDA.@sync EvoTrees.predict_gpu!($cache_g.pred_gpu, $tree, $cache_g.X_bin)

###########################
# Tree GPU
###########################
δ𝑤, K, edges, X_bin, nodes, out, left, right = cache_g.δ𝑤, cache_g.K, cache_g.edges, cache_g.X_bin, cache_g.nodes, cache_g.out, cache_g.left, cache_g.right;

# 9.613 ms (81 allocations: 13.55 KiB)
# 𝑗2 = CuArray(sample(UInt32.(1:100), 50, replace=false, ordered=true))
# @time EvoTrees.update_hist_gpu!(params_g.loss, nodes[1].h, δ𝑤, X_bin, 𝑖, 𝑗, K)
# println(nodes[1].h)
CUDA.@time EvoTrees.update_hist_gpu!(params_g.loss, nodes[1].h, δ𝑤, X_bin, 𝑖, 𝑗, K)
CUDA.@time EvoTrees.update_hist_gpu!(params_g.loss, nodes[1].h, δ𝑤, X_bin, 𝑖, 𝑗, K)

# @btime EvoTrees.update_hist_gpu!($params_g.loss, $nodes[1].h, $δ𝑤, $X_bin, $𝑖, $𝑗, $K)
# @btime EvoTrees.update_hist_gpu!($nodes[1].h, $δ𝑤, $X_bin, $nodes[1].𝑖, $𝑗)
# @code_warntype EvoTrees.update_hist_gpu!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)

# depth=1
# nid = 2^(depth - 1):2^(depth) - 1
# # 97.000 μs (159 allocations: 13.09 KiB)
# @time CUDA.@sync EvoTrees.update_gains_gpu!(gains::AbstractArray{T,3}, hist::AbstractArray{T,4}, histL::AbstractArray{T,4}, histR::AbstractArray{T,4}, 𝑗::AbstractVector{S}, params_g, nid, depth);
# @btime CUDA.@sync EvoTrees.update_gains_gpu!(gains::AbstractArray{T,3}, hist::AbstractArray{T,4}, histL::AbstractArray{T,4}, histR::AbstractArray{T,4}, 𝑗::AbstractVector{S}, params_g, nid, depth);
# gains[:,:,1]

# tree = EvoTrees.TreeGPU(UInt32(params_g.max_depth), model_g.K, params_g.λ)
# n = 1
# best = findmax(view(gains, :,:,n))
# if best[2][1] != params_g.nbins && best[1] > -Inf
#     tree.gain[n] = best[1]
#     tree.feat[n] = best[2][2]
#     tree.cond_bin[n] = best[2][1]
#     tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
# end
# tree.split[n] = tree.cond_bin[n] != 0

# # 673.900 μs (600 allocations: 29.39 KiB)
# @time CUDA.@sync EvoTrees.update_set_gpu!(𝑛, 𝑖, X_bin, tree.feat, tree.cond_bin, params_g.nbins)
# @btime CUDA.@sync EvoTrees.update_set_gpu!($𝑛, $𝑖, $X_bin, $tree.feat, $tree.cond_bin, $params_g.nbins)
