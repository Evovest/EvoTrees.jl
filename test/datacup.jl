using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample

using Revise
using EvoTrees
using EvoTrees: get_gain, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, predict, predict!, find_split_turbo!, SplitTrack, update_track!, sigmoid
using EvoTrees: get_edges, binarize, find_bags

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv")
names(data)

features = data[1:53]
X = convert(Array, features)
# X = X + randn(size(X)) * 0.0001
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

params1 = EvoTreeRegressor(
    loss=:logistic, metric=:logloss,
    nrounds=100, nbins=64,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=0.5, seed = 123)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10)
@time model = grow_gbtree(X_train, Y_train, params1, print_every_n = 1)
@time pred_train_linear = EvoTrees.predict(model, X_train)


# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
𝑤 = ones(size(X, 1))
pred = zeros(size(Y, 1))
# @time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
update_grads!(params1.loss, params1.α, pred, Y, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ), sum(δ²), sum(𝑤)
gain = get_gain(params1.loss, ∑δ, ∑δ², ∑𝑤, params1.λ)

# initialize train_nodes
train_nodes = Vector{TrainNode{Float64, BitSet, Array{Int64, 1}, Int}}(undef, 2^params1.max_depth-1)
for feat in 1:2^params1.max_depth-1
    train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, BitSet([0]), [0])
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

# manual check
x1 = edges[2]
x2 = [0, x1[1], 0.1, x1[2], 0.5, x1[3], 0.95, x1[4]]
x2_bin = searchsortedlast.(Ref(edges[2][1:end-1]), x2) .+ 1
x2_bag = find_bags(x2_bin)

function prep(X, params)
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)
    bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
    for feat in 1:size(𝑗, 1)
        bags[feat] = find_bags(X_bin[:,feat])
    end
    return bags
end

𝑖_set = BitSet(𝑖);
@time bags = prep(X, params1);

length(bags)
bins_length = map(length, bags)

for col in eachcol(X_bin)
    print(minimum(col))
end
max = Int[]
for col in eachcol(X_bin)
    push!(max, maximum(col))
end
max

feat = 1
typeof(bags[feat][1])
train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, BitSet(𝑖), 𝑗)
find_split_bitset!(bags[feat], δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1, splits[feat], tracks[feat], edges[feat], train_nodes[1].𝑖)
