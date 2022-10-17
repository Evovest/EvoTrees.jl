using Statistics
using StatsBase: sample, sample!
using EvoTrees
using BenchmarkTools
using CUDA

# prepare a dataset
features = rand(Int(1.25e4), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace=false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]


###########################
# Tree CPU
###########################
params_c = EvoTrees.EvoTreeMLE(T=Float32,
    loss=:logistic,
    nrounds=200,
    lambda=0.1, gamma=0.0, eta=0.1,
    max_depth=5, min_weight=1.0,
    rowsample=0.5, colsample=0.5, nbins=16);

model_c, cache_c = EvoTrees.init_evotree(params_c, x_train, y_train);
EvoTrees.grow_evotree!(model_c, cache_c)
p = model_c(x_train)
sort(p[:,1])
sort(p[:,2])

# initialize from cache
params_c = model_c.params
x_size = size(cache_c.X_bin)

# select random rows and cols
sample!(params_c.rng, cache_c.𝑖_, cache_c.nodes[1].𝑖, replace=false, ordered=true);
sample!(params_c.rng, cache_c.𝑗_, cache_c.𝑗, replace=false, ordered=true);
# @btime sample!(params_c.rng, cache_c.𝑖_, cache_c.nodes[1].𝑖, replace=false, ordered=true);
# @btime sample!(params_c.rng, cache_c.𝑗_, cache_c.𝑗, replace=false, ordered=true);

𝑖 = cache_c.nodes[1].𝑖
𝑗 = cache_c.𝑗

# build a new tree
# 897.800 μs (6 allocations: 736 bytes)
get_loss_type(m::EvoTreeGaussian{L,T,S}) where {L,T,S} = L
get_loss_type(m::EvoTrees.EvoTreeLogistic{L,T,S}) where {L,T,S} = L

L = get_loss_type(params_c)
@time EvoTrees.update_grads!(L, cache_c.δ𝑤, cache_c.pred, cache_c.Y; alpha=params_c.alpha)
cache_c.δ𝑤

sort(cache_c.δ𝑤[1, :])
sort(cache_c.δ𝑤[2, :])
sort(cache_c.δ𝑤[3, :])
sort(cache_c.δ𝑤[4, :])

p = collect(-3:0.5:3)
y = collect(-3:0.5:3)

function get_grads(p, y)
    grad = zeros(length(p), length(y))
    for i in eachindex(p)
        for j in eachindex(y)
            # alternate from 1
            # grad[i, j] = -(exp(-2s) * (u - y) * (u - y + exp(s) * sinh(exp(-s) * (u - y)))) / (1 + cosh(exp(-s) * (u - y)))
            grad[i, j] = (exp(-2 * p[i]) * (0.0 - y[j]) * (0.0 - y[j] + exp(p[i]) * sinh(exp(-p[i]) * (0.0 - y[j])))) / (1 + cosh(exp(-p[i]) * (0.0 - y[j])))
        end
    end
    return grad
end

grads = get_grads(p, y)
heatmap(grads)
# @btime EvoTrees.update_grads!($params_c.loss, $cache_c.δ𝑤, $cache_c.pred_cpu, $cache_c.Y_cpu, $params_c.α)
# ∑ = vec(sum(cache_c.δ[𝑖,:], dims=1))
# gain = EvoTrees.get_gain(params_c.loss, ∑, params_c.λ)
# assign a root and grow tree
# train_nodes[1] = EvoTrees.TrainNode(UInt32(0), UInt32(1), ∑, gain)

# 62.530 ms (7229 allocations: 17.43 MiB)
# 1.25e5: 9.187 ms (7358 allocations: 2.46 MiB)
tree = EvoTrees.Tree(params_c.max_depth, model_c.K, zero(typeof(params_c.λ)))
@time EvoTrees.grow_tree!(tree, cache_c.nodes, params_c, cache_c.δ𝑤, cache_c.edges, cache_c.𝑗, cache_c.left, cache_c.left, cache_c.right, cache_c.X_bin, cache_c.K)
@btime EvoTrees.grow_tree!($EvoTrees.Tree(params_c.max_depth, model_c.K, zero(typeof(params_c.λ))), $cache_c.nodes, $params_c, $cache_c.δ𝑤, $cache_c.edges, $cache_c.𝑗, $cache_c.left, $cache_c.left, $cache_c.right, $cache_c.X_bin, $cache_c.K)

@time EvoTrees.grow_tree!(EvoTrees.Tree(params_c.max_depth, model_c.K, params_c.λ), params_c, cache_c.δ, cache_c.hist, cache_c.histL, cache_c.histR, cache_c.gains, cache_c.edges, 𝑖, 𝑗, 𝑛, cache_c.X_bin);
@btime EvoTrees.grow_tree!(EvoTrees.Tree($params_c.max_depth, $model_c.K, $params_c.λ), $params_c, $cache_c.δ, $cache_c.hist, $cache_c.histL, $cache_c.histR, $cache_c.gains, $cache_c.edges, $𝑖, $𝑗, $𝑛, $cache_c.X_bin);
@code_warntype EvoTrees.grow_tree!(EvoTrees.Tree(params_c.max_depth, model_c.K, params_c.λ), params_c, cache_c.δ, cache_c.hist, cache_c.histL, cache_c.histR, cache_c.gains, cache_c.edges, 𝑖, 𝑗, 𝑛, cache_c.X_bin);

# push!(model_c.trees, tree)
# 1.883 ms (83 allocations: 13.77 KiB)
@btime EvoTrees.predict!(model_c.params.loss, cache_c.pred_cpu, tree, cache_c.X, model_c.K)

δ𝑤, K, edges, X_bin, nodes, out, left, right = cache_c.δ𝑤, cache_c.K, cache_c.edges, cache_c.X_bin, cache_c.nodes, cache_c.out, cache_c.left, cache_c.right;

# 9.613 ms (81 allocations: 13.55 KiB)
# 1.25e5: 899.200 μs (81 allocations: 8.22 KiB)
@time EvoTrees.update_hist!(params_c.loss, nodes[1].h, δ𝑤, X_bin, 𝑖, 𝑗, K)
@btime EvoTrees.update_hist!($params_c.loss, $nodes[1].h, $δ𝑤, $X_bin, $𝑖, $𝑗, $K)
@btime EvoTrees.update_hist!($nodes[1].h, $δ𝑤, $X_bin, $nodes[1].𝑖, $𝑗)
@code_warntype EvoTrees.update_hist!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)

j = 1
# 8.399 μs (80 allocations: 13.42 KiB)
n = 1
nodes[1].∑ .= vec(sum(δ𝑤[:, 𝑖], dims=2))
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

# 204.900 μs (1 allocation: 96 bytes)
offset = 0
@time EvoTrees.split_set!(left, right, 𝑖, X_bin, tree.feat[n], tree.cond_bin[n], offset)
@btime EvoTrees.split_set!($left, $right, $𝑖, $X_bin, $tree.feat[n], $tree.cond_bin[n], $offset)
@code_warntype EvoTrees.split_set!(left, right, 𝑖, X_bin, tree.feat[n], tree.cond_bin[n])

# 1.25e5: 227.200 μs (22 allocations: 1.44 KiB)
@time EvoTrees.split_set_threads!(out, left, right, 𝑖, X_bin, tree.feat[n], tree.cond_bin[n], offset)
@btime EvoTrees.split_set_threads!($out, $left, $right, $𝑖, $X_bin, $tree.feat[n], $tree.cond_bin[n], $offset, Int(2e15))
