using RDatasets
using DataFrames
using Statistics
using GBT

using Base.Threads: @threads
using GBT: get_max_gain, grow_tree, grow_gbt, SplitInfo, TreeLeaf, Params, predict

# prepare a dataset
iris = dataset("datasets", "iris")
names(iris)

features = iris[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]]
X = convert(Array, features)
Y = iris[:Species]
Y = Y .== "virginica"

# identify best split
# iter on each variable
# compute gain at each split point

# initial info
δ, δ² = GBT.grad_hess(zeros(size(Y,1)), Y)
∑δ, ∑δ² = sum(δ), sum(δ²)

# set parameters
nrounds = 2
λ = 0.0001
γ = 1e-3
η = 0.1
max_depth = 3
min_weight = 5.0
params1 = Params(nrounds, λ, γ, η, max_depth, min_weight)

gain_ini = GBT.get_gain(∑δ, ∑δ², params1.λ)
best = GBT.SplitInfo(gain_ini, 0.0, 0.0, ∑δ, ∑δ², -Inf, -Inf, Vector[], Vector[], 0, 0.0)
find_split_1 = GBT.find_split(X[:, 1], δ, δ², ∑δ, ∑δ², params1.λ) # returns gain value and idx split

splits = Vector{SplitInfo}(undef, size(X, 2))
# Search best split for each feature - to be multi-threaded
@threads for feat in 1:size(X, 2)
    splits[feat] = GBT.find_split(X[:, feat], δ, δ², ∑δ, ∑δ², params1.λ)
    splits[feat].feat = feat # returns gain value and idx split
end
best = get_max_gain(splits)

root = TreeLeaf(1, ∑δ, ∑δ², gain_ini, 0.0)
params1 = Params(100, λ, γ, 0.5, 3, min_weight)
tree = grow_tree(root, X, δ, δ², params1)

# predict - map a sample to tree-leaf prediction
pred = predict(tree, X)
mean((pred .- Y) .^ 2)
println(sort(unique(predicts)))

# build model
model = grow_gbt(X, Y, params1)
pred = predict(model, X)
println(sort(unique(pred)))
mean((pred .- Y) .^ 2)

pred1 = predict(model.trees[1], X)
println(sort(unique(pred1)))

pred2 = predict(model.trees[2], X)
println(sort(unique(pred2)))

pred3 = predict(model.trees[3], X)
println(sort(unique(pred3)))

# get the positions indices from ascending order
# only to be called once before stacking the trees
X_sortperm = mapslices(sortperm, X, dims = 1)

X_sortperm[:,1]
X[:,4][X_sortperm[:,4]]


# Threading
using BenchmarkTools
Threads.nthreads()

function threads_sum()
    M = 1000000
    a=Vector{Float64}(undef, M)
    Threads.@threads for i=1:M
        a[i]=log1p(i)
    end#for
    return a
end
