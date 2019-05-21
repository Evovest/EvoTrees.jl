using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample
using StaticArrays
using Revise
using BenchmarkTools
using EvoTrees
using EvoTrees: get_gain, get_edges, binarize, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, SplitTrack, Tree, TrainNode, TreeNode, Params, predict, predict!, sigmoid
using EvoTrees: scan, find_bags, find_bags_direct, scan, find_histogram, intersect_test, update_bags!, update_bags_intersect

# prepare a dataset
features = rand(100_000, 100)

X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# set parameters
loss = :linear
nrounds = 10
λ = 1.0
γ = 0.0
η = 0.5
max_depth = 5
min_weight = 5.0
rowsample = 1.0
colsample = 1.0
nbins = 32
# params1 = Params(nrounds, λ, γ, η, max_depth, min_weight, :linear)
params1 = Params(:linear, nrounds, λ, γ, 1.0, 5, min_weight, rowsample, colsample, nbins)

# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
𝑤 = ones(size(X, 1))
pred = zeros(size(Y, 1))
# @time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
update_grads!(Val{params1.loss}(), pred, Y, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ), sum(δ²), sum(𝑤)
gain = get_gain(∑δ, ∑δ², ∑𝑤, params1.λ)

# initialize train_nodes
train_nodes = Vector{TrainNode{Float64, BitSet, Array{Int64, 1}, Int}}(undef, 2^params1.max_depth-1)
for feat in 1:2^params1.max_depth-1
    train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, BitSet([0]), [0])
    # train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, Set([0]), [0], bags)
end

# initializde node splits info and tracks - colsample size (𝑗)
splits = Vector{SplitInfo{Float64, Int}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    splits[feat] = SplitInfo{Float64, Int}(-Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, feat, 0.0)
end
tracks = Vector{SplitTrack{Float64}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    tracks[feat] = SplitTrack{Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
end

@time edges = get_edges(X, params1.nbins)
@time X_bin = binarize(X, edges)

bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    bags[feat] = find_bags(X_bin[:,feat])
end

function prep1(X, params)
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)
    bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
    for feat in 1:size(𝑗, 1)
        bags[feat] = find_bags(X_bin[:,feat])
    end
    return bags
end

function prep2(X, params)
    edges = get_edges(X, params.nbins)
    bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
    for feat in 1:size(𝑗, 1)
        bags[feat] = find_bags_direct(X[:,feat], edges[feat])
    end
    return bags
end

@time bags = prep1(X_train, params1);
@time bags = prep2(X_train, params1);

@time train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, BitSet(𝑖), 𝑗)
@time tree = grow_tree(bags, δ, δ², 𝑤, params1, train_nodes, splits, tracks, edges)
@btime tree = grow_tree($bags, $δ, $δ², $𝑤, $params1, $train_nodes, $splits, $tracks, $edges)

params1 = Params(:linear, 5, λ, γ, 1.0, 5, min_weight, rowsample, colsample, nbins)
@btime model = grow_gbtree($X_train, $Y_train, $params1, print_every_n = 1, metric=:mae)
@time pred_train = predict(model, X_train)

params1 = Params(:linear, 10, λ, γ, 0.1, 5, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 1, metric=:mae)
@btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 1, metric=:mae)

@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))

#############################################
# Quantiles with Sets
#############################################

𝑖_set = BitSet(𝑖);

feat = 9
find_histogram(bags[feat], δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1.λ, splits[feat], tracks[feat], edges[feat], train_nodes[feat].𝑖)
@time find_histogram(bags[1], δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1.λ, splits[1], tracks[1], edges[1], train_nodes[1].𝑖)
@btime find_histogram($bags[1], $δ, $δ², $𝑤, $∑δ, $∑δ², $∑𝑤, $params1.λ, $splits[1], $tracks[1], $edges[1], $train_nodes[1].𝑖)



new_bags = Vector{Vector{BitSet}}(undef, length(bags))
for i in 1:length(new_bags)
    new_bags[i] = Vector{BitSet}(undef, length(bags[i]))
    for j in 1:length(bags[i])
        new_bags[i][j] = BitSet()
    end
end

length(union(train_nodes[1].bags[1][1:13]...))
length(union(train_nodes[1].bags[1][1:13]...))
length(new_bags[2])
length(new_bags[2][1])
length(bags[2][32])
typeof(bags)
@btime update_bags_intersect($new_bags, $bags, $union(train_nodes[1].bags[1][1:13]...))
length(new_bags[2])
length(new_bags[2][2])
length(bags[2][1])

# extract the best feat from bags, and join all the underlying bins up to split point
best_bag = bags[1]
bins_L = union(best_bag[1:4]...)

function set_1(x, y)
    intersect!(x, y)
    return x
end

x = rand(UInt32, 100_000)
y = rand(x, 1000)

x_set = BitSet(x);
y_set = BitSet(y);

@btime set_1(x, y)
@btime set_1(x_set, y)


x = rand([1,2,3,4,5,6,7,8,9,10, 11,12], 1000)
x = rand(1000)
x_edges = quantile(x, (0:10)/10)
x_edges = unique(x_edges)
x_edges = x_edges[2:(end-1)]

length(x_edges)

x_bin = searchsortedlast.(Ref(x_edges), x) .+ 1
using StatsBase
x_map = countmap(x_bin)

x = reshape(x, (1000, 1))
x_edges = get_edges(x)
unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:(end-1)]
x_bin = searchsortedlast.(Ref(x_edges[1]), x[:,1]) .+ 1
x_map = countmap(x_bin)

edges = get_edges(X, 32)
X_bin = zeros(UInt8, size(X))
@btime binindices(X[:,1], edges[1])
@btime X_bin = binarize(X, edges)

using StatsBase
x_map = countmap(x_bin)

x_edges[1]
