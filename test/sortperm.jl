using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample

using EvoTrees
using EvoTrees: get_gain, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
# prepare a dataset
features = rand(UInt8, 100000, 300)
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

# idx
X_perm = zeros(Int, size(X))
@threads for feat in 1:size(X, 2)
    X_perm[:, feat] = sortperm(X[:, feat]) # returns gain value and idx split
    # idx[:, feat] = sortperm(view(X, :, feat)) # returns gain value and idx split
end

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
params1 = Params(:linear, 1, λ, γ, 1.0, 2, min_weight, rowsample, colsample)

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

# placeholder for sort perm
perm_ini = zeros(Int, size(X))
for feat in 1:size(𝑗, 1)
    perm_ini[:, feat] = 𝑖
end

train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)

function sortperm_test1(node, splits, perm_ini, params)
    node_size = size(node.𝑖, 1)
    @threads for feat in node.𝑗
        sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = true)
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])
    end
end

function sortperm_test2(node, splits, perm_ini, params)
    node_size = size(node.𝑖, 1)
    @threads for feat in node.𝑗
        perm_ini[1:node_size, feat] = sortperm(X[node.𝑖, feat], alg = QuickSort)
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])
    end
end

function sortperm_radix_test1(node, splits, perm_ini, params)
    node_size = size(node.𝑖, 1)
    @threads for feat in node.𝑗
        perm_ini[1:node_size, feat] = faster_sortperm_radix(X[node.𝑖, feat])
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])
    end
end

@time sortperm_test1(train_nodes[1], splits, perm_ini, params1)
@time sortperm_test2(train_nodes[1], splits, perm_ini, params1)
@time sortperm_radix_test1(train_nodes[1], splits, perm_ini, params1)

sizeof(X)/1024^2

using SortingAlgorithms
function faster_sortperm_radix(v)
  ai = [Pair(i, a) for (i,a) in enumerate(v)]
  sort!(ai, by=x->x.second, alg=RadixSort)
  [a.first for a in ai]
end

function faster_sortperm(v)
  ai = [Pair(i, a) for (i,a) in enumerate(v)]
  sort!(ai, by=x->x.second)
  [a.first for a in ai]
end


function test_faster_sortperm(node, splits, perm_ini, params)
    node_size = size(node.𝑖, 1)
    @threads for feat in node.𝑗
        perm_ini[1:node_size, feat] = faster_sortperm(X[node.𝑖, feat])
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])

    end
end

function test_faster_sortperm_radix(node, splits, perm_ini, params)
    node_size = size(node.𝑖, 1)
    @threads for feat in node.𝑗
        perm_ini[1:node_size, feat] = faster_sortperm_radix(X[node.𝑖, feat])
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])

    end
end

@time test_faster_sortperm(train_nodes[1], splits, perm_ini, params1)
@time test_faster_sortperm_radix(train_nodes[1], splits, perm_ini, params1)

x1 = rand(Int, 100000)

@time sortperm(x1)
@time faster_sortperm(x1)
@time faster_sortperm_radix(x1)
