using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample

using Revise
using Traceur
using EvoTrees
using EvoTrees: get_gain, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Array, features)
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

# params1 = Params(nrounds, λ, γ, η, max_depth, min_weight, :linear)
params1 = Params(:linear, 1, λ, γ, 1.0, 5, min_weight, rowsample, colsample)

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

x = X[:, 5]
x_sortperm = sortperm(x)
x_sort = x[x_sortperm]
δ_sort = δ[x_sortperm]
δ²_sort = δ²[x_sortperm]

@time find_split!(x_sort, δ_sort, δ²_sort, 𝑤, ∑δ, ∑δ², ∑𝑤, params1.λ, splits[1], tracks[1])
@code_warntype find_split!(x_sort, δ_sort, δ²_sort, ∑δ, ∑δ², params1.λ, splits[1], tracks[1])
