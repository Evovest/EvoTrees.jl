using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample, quantile
using EvoTrees
using EvoTrees: sigmoid, logit
using Plots
using GraphRecipes

# prepare a dataset
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(
    loss=:linear, metric=:mse,
    nrounds=100, nbins = 100,
    λ = 0.5, γ=0.1, η=0.05,
    max_depth = 4, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = predict(model, X_train)
# @time pred_eval_linear = predict(model, X_eval)
# mean(abs.(pred_train_linear .- Y_train))
# sqrt(mean((pred_train_linear .- Y_train) .^ 2))

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")


params1 = EvoTreeRegressor(
    loss=:linear, metric=:mse,
    nrounds=100, nbins = 100,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 2, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

anim = @animate for i=1:20
    params1.nrounds = (i-1)*5+1
    model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = Inf)
    pred_train_linear = predict(model, X_train)
    x_perm = sortperm(X_train[:,1])
    plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
    plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
end
gif(anim, "anim_fps1.gif", fps = 1)

# plot tree
function treemat(tree)
    mat = zeros(Int, length(tree), length(tree))
    for i in 1:length(tree)
        if tree[i].split
            mat[i,tree[i].left] = 1
            mat[tree[i].left, i] = 1
            mat[i,tree[i].right] = 1
            mat[tree[i].right, i] = 1
        end
        mat = sparse(mat)
    end
    return mat
end

# plot tree
function nodenames(tree)
    names = []
    for i in 1:length(tree)
        if tree[i].split
            push!(names, "feat: " * string(tree[i].feat) * "\n< " * string(round(tree[i].cond, sigdigits=3)))
        else
            push!(names, "pred: " * string(round(tree[i].pred[1], sigdigits=3)))
        end
    end
    return names
end

tree1 = model.trees[2].nodes
mat = treemat(tree1)
nodes = nodenames(tree1)
p = graphplot(mat, method=:tree, node_weights=ones(length(tree1)) .* 10, names = nodes, linecolor=:grey, nodeshape=:ellipse, nodecolor="#66ffcc")
savefig(p, "tree1.svg")
