using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample
using StaticArrays
using Revise
using BenchmarkTools
using EvoTrees
using EvoTrees: get_gain, get_edges, binarize, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, EvoTreeRegressor, predict, predict!, sigmoid
using EvoTrees: find_bags, update_bags!, find_split_static!, pred_leaf, sigmoid, logit
using Plots
using Distributions

# prepare a dataset
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]
𝑖 = collect(1:size(X_train,1))

# set parameters
params1 = EvoTreeRegressor(
    loss=:gaussian, metric=:gaussian,
    nrounds=100, nbins=100,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0, seed=123)

mean(Y_train.^2)

# initial info
@time δ, δ² = zeros(SVector{params1.K, Float64}, size(X_train, 1)), zeros(SVector{params1.K, Float64}, size(X_train, 1))
𝑤 = zeros(SVector{1, Float64}, size(X_train, 1)) .+ 1
pred = zeros(SVector{params1.K,Float64}, size(X_train,1))
@time update_grads!(params1.loss, params1.α, pred, Y_train, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
@time gain = get_gain(params1.loss, ∑δ, ∑δ², ∑𝑤, params1.λ)
# @btime gain = get_gain($params1.loss, $∑δ, $∑δ², $∑𝑤, $params1.λ)

# initialize train_nodes
train_nodes = Vector{TrainNode{params1.K, Float64, BitSet, Array{Int64, 1}, Int}}(undef, 2^params1.max_depth-1)
for node in 1:2^params1.max_depth-1
    train_nodes[node] = TrainNode(0, SVector{params1.K, Float64}(fill(-Inf, params1.K)), SVector{params1.K, Float64}(fill(-Inf, params1.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, BitSet([0]), [0])
    # train_nodes[feat] = TrainNode(0, fill(-Inf, params1.K), fill(-Inf, params1.K), -Inf, -Inf, BitSet([0]), [0])
end

# initializde node splits info and tracks - colsample size (𝑗)
splits = Vector{SplitInfo{params1.K, Float64, Int}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    splits[feat] = SplitInfo{params1.K, Float64, Int}(gain, SVector{params1.K, Float64}(zeros(params1.K)), SVector{params1.K, Float64}(zeros(params1.K)), SVector{1, Float64}(zeros(1)), SVector{params1.K, Float64}(zeros(params1.K)), SVector{params1.K, Float64}(zeros(params1.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
end

# binarize data and create bags
@time edges = get_edges(X_train, params1.nbins)
@time X_bin = binarize(X_train, edges)
@time bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
function prep(X_bin, bags)
    @threads for feat in 1:size(𝑗, 1)
         bags[feat] = find_bags(X_bin[:,feat])
    end
    return bags
end
@time bags = prep(X_bin, bags)

# initialize histograms
feat=1
hist_δ = Vector{Vector{SVector{params1.K, Float64}}}(undef, size(𝑗, 1))
hist_δ² = Vector{Vector{SVector{params1.K, Float64}}}(undef, size(𝑗, 1))
hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    hist_δ[feat] = zeros(SVector{params1.K, Float64}, length(bags[feat]))
    hist_δ²[feat] = zeros(SVector{params1.K, Float64}, length(bags[feat]))
    hist_𝑤[feat] = zeros(SVector{1, Float64}, length(bags[feat]))
end

# grow single tree
#  0.135954 seconds (717.54 k allocations: 15.219 MiB)
@time train_nodes[1] = TrainNode(1, SVector(∑δ), SVector(∑δ²), SVector(∑𝑤), gain, BitSet(𝑖), 𝑗)
@time tree = grow_tree(bags, δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, params1, train_nodes, splits, edges, X_bin)
# @btime tree = grow_tree($bags, $δ, $δ², $𝑤, $hist_δ, $hist_δ², $hist_𝑤, $params1, $train_nodes, $splits, $tracks, $edges, $X_bin)
@time pred_train = predict(tree, X_train, params1.K)
# 705.901 μs (18 allocations: 626.08 KiB)
# @btime pred_train = predict($tree, $X_train, $params1.K)
@time pred_leaf_ = pred_leaf(params1.loss, train_nodes[1], params1, δ²)
# 1.899 ns (0 allocations: 0 bytes)
# @btime pred_leaf_ = pred_leaf($params1.loss, $train_nodes[1], $params1, $δ²)
# @btime pred_train = predict($tree, $X_train, params1.K)

@time model = grow_gbtree(X_train, Y_train, params1, print_every_n = 20)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, print_every_n = 1)
@time pred_train = predict(model, X_train)
# @btime pred_train = predict($model, $X_train)

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train[:,1][x_perm], color = "navy", linewidth = 1.5, label = "Median")
# σ²
plot!(X_train[:,1][x_perm], sqrt.(pred_train[:,2][x_perm]), color = "red", linewidth = 1.5, label = "sigma")
# q20
dist = Normal.(pred_train[:,1][x_perm], sqrt.(pred_train[:,2][x_perm]))
pred_train_q20 = quantile.(dist, 0.2)
pred_train_q80 = quantile.(dist, 0.8)
sum(pred_train_q20 .< Y_train[x_perm]) / size(Y_train,1)
sum(pred_train_q80 .< Y_train[x_perm]) / size(pred_train,1)
plot!(X_train[:,1][x_perm], pred_train_q20, color = "green", linewidth = 1.5, label = "Q20")
plot!(X_train[:,1][x_perm], pred_train_q80, color = "green", linewidth = 1.5, label = "Q80")
savefig("gaussian_likelihood.png")
