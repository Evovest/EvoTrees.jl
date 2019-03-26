using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample

using Revise
using EvoTrees
using EvoTrees: get_gain, update_gains!, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Array, features)
Y = data[54]
Y = convert(Array{Float64}, Y)
𝑖 = collect(1:size(X,1))

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
pred = zeros(size(Y, 1))
@time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
# @code_warntype update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
∑δ, ∑δ² = sum(δ), sum(δ²)

gain = get_gain(∑δ, ∑δ², params1.λ)
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# initialize train_nodes
train_nodes = Vector{TrainNode}(undef, 2^params1.max_depth-1)
for feat in 1:2^params1.max_depth-1
    train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, [0], [0])
end

root = TrainNode(1, ∑δ, ∑δ², gain, 𝑖, 𝑗)
train_nodes[1] = root
@time tree = grow_tree(X, δ, δ², params1, perm_ini, train_nodes)

# predict - map a sample to tree-leaf prediction
@time pred = predict(tree, X)
# pred = sigmoid(pred)
sqrt(mean((pred .- Y) .^ 2))


# prediction from single tree - assign each observation to its final leaf
function predict_1(tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(Float64, size(X, 1))
    for i in 1:size(X, 1)
        id = Int(1)
        while tree.nodes[id].split
            if X[i, tree.nodes[id].feat] <= tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
    return pred
end

function predict_1!(pred, tree::Tree, X::AbstractArray{T, 2}) where T<:Real

    for i in 1:size(X, 1)
        id = Int(1)
        while tree.nodes[id].split
            if X[i, tree.nodes[id].feat] <= tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
    return pred
end

@time pred = predict_1(tree, X)
@code_warntype predict_1(tree, X)

pred = zeros(Float32, size(X, 1))
@code_warntype predict_1!(pred, tree, X)
@time pred = predict_1!(pred, tree, X)
@time pred = predict_2!(pred, tree, X)

sizeof(pred)/1024


mean((pred .- Y) .^ 2)


# prediction from single tree - assign each observation to its final leaf
function predict_2(tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(size(X, 1))
    for i in size(X, 1)
        pred[i] += tree.nodes[30].pred
    end
    return pred
end

@time pred = predict_2(tree, X)

sizeof(X)/1000
sizeof(pred)/1000
