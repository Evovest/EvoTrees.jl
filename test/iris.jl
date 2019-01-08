using DataFrames
using CSV
using Statistics
using GBT

using Base.Threads: @threads
using GBT: get_gain, update_gains!, get_max_gain, update_grads!, grow_tree!, grow_gbtree, SplitInfo2, Tree, Node, Params, predict, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
iris = CSV.read("./data/iris.csv", allowmissing = :auto)
names(iris)

features = iris[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]]
X = convert(Array, features)
Y = iris[:Species]
Y = Y .== "virginica"

# initial info
δ, δ² = grad_hess(zeros(size(Y,1)), Y)
∑δ, ∑δ² = sum(δ), sum(δ²)

# set parameters
nrounds = 2
λ = 0.0001
γ = 1e-3
η = 0.1
max_depth = 3
min_weight = 5.0
params1 = Params(nrounds, λ, γ, η, max_depth, min_weight)

gain = get_gain(∑δ, ∑δ², params1.λ)
root = TreeLeaf(1, ∑δ, ∑δ², gain, 0.0)
params1 = Params(100, λ, γ, 0.1, 5, min_weight)
tree = grow_tree(root, X, δ, δ², params1)

# predict - map a sample to tree-leaf prediction
pred = predict(tree, X)
mean((pred .- Y) .^ 2)
println(sort(unique(pred)))

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
