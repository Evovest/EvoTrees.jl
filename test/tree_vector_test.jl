using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using BenchmarkTools
# using Profile
using StatsBase: sample

using Revise
# using Traceur
using EvoTrees
using EvoTrees: get_gain, get_max_gain, update_grads!, eval_metric, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Matrix, features)
Y = data[54]
Y = convert(Array{Float64}, Y)
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# idx
X_perm = zeros(Int, size(X))
@threads for feat in 1:size(X, 2)
    X_perm[:, feat] = sortperm(X[:, feat]) # returns gain value and idx split
    # idx[:, feat] = sortperm(view(X, :, feat)) # returns gain value and idx split
end

# placeholder for sort perm
perm_ini = zeros(Int, size(X))

# set parameters
nrounds = 1
λ = 1.0
γ = 1e-15
η = 0.5
max_depth = 5
min_weight = 5.0
rowsample = 1.0
colsample = 1.0
nbins = 128

# params1 = Params(nrounds, λ, γ, η, max_depth, min_weight, :linear)
params1 = Params(:linear, 1, λ, γ, 1.0, 2, min_weight, rowsample, colsample, nbins)

# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
𝑤 = ones(size(X, 1))
pred = zeros(size(Y, 1))
# @time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
update_grads!(Val{params1.loss}(), pred, Y, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ), sum(δ²), sum(𝑤)

gain = get_gain(∑δ, ∑δ², ∑𝑤, params1.λ)

# initialize train_nodes
train_nodes = Vector{TrainNode{Float64, Array{Int64,1}, Array{Int64, 1}, Int}}(undef, 2^params1.max_depth-1)
for feat in 1:2^params1.max_depth-1
    train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, [0], [0])
end
# initializde node splits info and tracks - colsample size (𝑗)
splits = Vector{SplitInfo{Float64, Int}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    splits[feat] = SplitInfo{Float64, Int}(-Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
end
tracks = Vector{SplitTrack{Float64}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    tracks[feat] = SplitTrack{Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
end

root = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
train_nodes[1] = root

tree = [TreeNode(1.3)]
Tree(tree)
tree = Vector{TreeNode{Float64, Int, Bool}}()

@time tree = grow_tree(X, δ, δ², 𝑤, params1, perm_ini, train_nodes, splits, tracks)
# @code_warntype grow_tree(X, δ, δ², 𝑤, params1, perm_ini, train_nodes, splits, tracks)

# predict - map a sample to tree-leaf prediction
# @time pred = predict(tree, X)
@time pred = predict(tree, X)
# @code_warntype predict(tree, X)

# pred = sigmoid(pred)
(mean((pred .- Y) .^ 2))
# println(sort(unique(pred)))

function test_grow(n, X, δ, δ², 𝑤, perm_ini, params)
    for i in 1:n
        root = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        train_nodes[1] = root
        grow_tree(X, δ, δ², 𝑤, params, perm_ini, train_nodes, splits, tracks)
        # grow_tree!(tree, view(X, :, :), view(δ, :), view(δ², :), params1)
    end
end

@time test_grow(1, X, δ, δ², 𝑤, perm_ini, params1)
# @time test_grow(100, X, δ, δ², 𝑤, perm_ini, params1)

# full model
params1 = Params(:linear, 1, λ, γ, 1.0, 5, min_weight, 1.0, 1.0)
@time model = grow_gbtree(X, Y, params1)
# model = grow_gbtree(X, Y, params1)

# predict - map a sample to tree-leaf prediction
# @time pred = predict(model, X)
@time pred = predict(model, X)
# @code_warntype predict(model, X)

# pred = sigmoid(pred)
sqrt(mean((pred .- Y) .^ 2))


# train model
params1 = Params(:linear, 100, 0.0, 0.0, 0.1, 6, 1.0, 0.5, 0.5, 250)
@btime model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, metric = :mse, print_every_n=20, early_stopping_rounds=100)

@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))
pred_eval = predict(model, X_eval)
mean((pred_eval .- Y_eval) .^ 2)

# train model
params1 = Params(:logistic, 100, 0.0, 0.0, 0.1, 5, 1.0, 0.5, 0.5)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, metric = :logloss, print_every_n=10, early_stopping_rounds=100)
@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))

####################################################
### Pred on binarised data
####################################################

X_bin = mapslices(x -> round.(31 .* (x .- minimum(x)) / (maximum(x) - minimum(x))), X, dims = 2)
X_bin = convert(Array{UInt8}, X_bin)
X_train_bin = convert(Array{UInt8}, round.(X_train*31))
X_eval_bin = convert(Array{UInt8}, round.(X_eval*31))

params1 = Params(:linear, 100, 0.0, 0.0, 0.1, 5, 1.0, 0.5, 0.5)
@time model = grow_gbtree(X_train_bin, Y_train, params1, X_eval = X_eval_bin, Y_eval = Y_eval, metric = :mse, print_every_n=20, early_stopping_rounds=100)
# model = grow_gbtree(X_train_bin, Y_train, params1, X_eval = X_eval_bin, Y_eval = Y_eval)

# predict - map a sample to tree-leaf prediction
pred = predict(model, X_eval_bin)
mean((pred .- Y_eval) .^ 2)

@time pred = predict(model, X_train_bin)
mean((pred .- Y_train) .^ 2)

params1 = Params(:logistic, 100, 0.0, 1.0, 0.1, 6, 1.0, 0.5, 0.5)
@time model = grow_gbtree(X_train_bin, Y_train, params1, X_eval = X_eval_bin, Y_eval = Y_eval, metric = :logloss, print_every_n=10, early_stopping_rounds=100)
@time pred = predict(model, X_train_bin)
mean((pred .- Y_train) .^ 2)

minimum(pred)
maximum(pred)

# big data test
X_train_bin2 = hcat(X_train_bin, X_train_bin, X_train_bin, X_train_bin)
X_train_bin2 = vcat(X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2, X_train_bin2)
Y_train2 = vcat(Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train, Y_train)
params1 = Params(:logistic, 10, 0.0, 1.0, 0.1, 6, 1.0, 0.5, 0.5)
@time model = grow_gbtree(X_train_bin2, Y_train2, params1, metric = :logloss, print_every_n=5)
