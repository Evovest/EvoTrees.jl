using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample, quantile
using Plots
using Plots: colormap
using Base.Threads: @threads

using Revise
using EvoTrees
using EvoTrees: sigmoid, logit
using EvoTrees: softmax
using EvoTrees: update_grads!, get_gain, TrainNode, SplitInfo, get_edges, binarize, find_bags, grow_tree, find_split_static!
using EvoTrees: pred_leaf, softmax
using Flux: onehot

# prepare a dataset
iris = CSV.read("./data/iris.csv")
names(iris)

features = iris[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]]
X = convert(Matrix, features)
Y = iris[:Species]
values = sort(unique(Y))
dict = Dict{String, Int}(values[i] => i for i in 1:length(values))
Y = map((x) -> dict[x], Y)

# train-eval split
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]
𝑖 = collect(1:size(X_train,1))
# scatter(X_train[:,1], X_train[:,2], color=Y_train, legend=nothing)
# scatter(X_eval[:,1], X_eval[:,2], color=Y_eval, legend=nothing)

##################################
# Step by step development
##################################
# set parameters
params1 = EvoTreeRegressor(
    loss=:softmax, metric=:mlogloss,
    nrounds=1, nbins=16,
    λ = 0.0, γ=0.0, η=0.3,
    max_depth = 3, min_weight = 1.0,
    rowsample=1.0, colsample=1.0,
    K = 3, seed=44)

# initial info
@time δ, δ² = zeros(SVector{params1.K, Float64}, size(X_train, 1)), zeros(SVector{params1.K, Float64}, size(X_train, 1))
𝑤 = zeros(SVector{1, Float64}, size(X_train, 1)) .+ 1
pred = zeros(SVector{params1.K,Float64}, size(X_train,1))
@time update_grads!(params1.loss, params1.α, pred, Y_train, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
@time gain = get_gain(params1.loss, ∑δ, ∑δ², ∑𝑤, params1.λ)

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
@time train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, BitSet(𝑖), 𝑗)
@time pred_leaf_ = pred_leaf(params1.loss, train_nodes[1], params1, δ²)
# @btime pred_leaf_ = pred_leaf($params1.loss, $train_nodes[1], $params1, $δ²)
@time tree = grow_tree(bags, δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, params1, train_nodes, splits, edges, X_bin)

# feat = 1
# typeof(bags[feat][1])
# train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, BitSet(𝑖), 𝑗)
# find_split_turbo!(bags[feat], view(X_bin,:,feat), δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1, splits[feat], tracks[feat], edges[feat], train_nodes[1].𝑖)

pred = predict(tree, X_train, params1.K)
for i in eachindex(pred)
    pred[i] = exp.(pred[i]) / sum(exp.(pred[i]))
end
pred_int = zeros(Int, length(pred))
for i in eachindex(pred)
    pred_int[i] = findmax(pred[i])[2]
end
sum(pred_int .== Y_train)

params1 = EvoTreeRegressor(
    loss=:softmax, metric=:mlogloss,
    nrounds=20, nbins=16,
    λ = 0.0, γ=1e-5, η=0.3,
    max_depth = 3, min_weight = 1.0,
    rowsample=1.0, colsample=1.0,
    K = 3, seed=44)

@time model = grow_gbtree(X_train, Y_train, params1, print_every_n = Inf)
# @time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 1)

sum(Y_train.==3)/length(Y_train)
@time pred_train = predict(model, X_train)
@time pred_eval = predict(model, X_eval)

maximum(pred_train)
minimum(pred_train)

pred_train_int = zeros(Int, length(Y_train))
for i in 1:size(pred_train, 1)
    pred_train_int[i] = findmax(pred_train[i,:])[2]
end

pred_eval_int = zeros(Int, length(Y_eval))
for i in 1:size(pred_eval, 1)
    pred_eval_int[i] = findmax(pred_eval[i,:])[2]
end
sum(pred_train_int .== Y_train), sum(pred_eval_int .== Y_eval)
