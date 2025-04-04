using Statistics
using StatsBase: sample, quantile
using EvoTrees
using EvoTrees: sigmoid, logit
using Plots
using GraphRecipes
using Random: seed!
using StaticArrays

# prepare a dataset
seed!(123)
X = rand(1_000, 2) .* 2
Y = sin.(X[:,1] .* π) .+ X[:,2]
Y = Y .+ randn(size(Y)) .* 0.1 #logit(Y)
# Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))

# make a grid
grid_size = 101
range = 2
X_grid = zeros(grid_size^2,2)
for j in 1:grid_size
    for i in 1:grid_size
        X_grid[grid_size*(j-1) + i,:] .= [(i-1) / (grid_size-1) * range,  (j-1) / (grid_size-1) * range]
    end
end
Y_grid = sin.(X_grid[:,1] .* π) .+ X_grid[:,2]

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# linear
params1 = EvoTreeRegressor(T=Float64,
    loss=:linear, metric=:mse,
    nrounds=100, nbins = 16,
    λ=0.0, γ=0.0, η=0.05,
    max_depth = 3, min_weight = 1.0,
    rowsample=0.8, colsample=1.0)

edges = EvoTrees.get_edges(X_train, params1.nbins)
X_bin = EvoTrees.binarize(X_train, edges)
@time model = model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
using BSON: @save
@save "blog/model_linear.bson" model

# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 25, metric=:mae)
@time pred_train_linear = predict(model, X_train)
# @time pred_eval_linear = predict(model, X_eval)
# mean(abs.(pred_train_linear .- Y_train))
# sqrt(mean((pred_train_linear .- Y_train) .^ 2))

μ = mean(Y_train)

x_perm_1 = sortperm(X_train[:,1])
x_perm_2 = sortperm(X_train[:,2])
p1 = plot(X_train[:,1], Y_train, ms = 3, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var1"), yaxis = ("target"), leg = false, cbar=true)
p1_bin = plot(X_bin[:,1], Y_train, ms = 3, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msw=0, background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("var1"), yaxis = ("target"), legend = false, cbar=true, label = "")
plot!(fill(μ, params1.nbins), lw=3, color="#66ffcc", background_color = "white", seriestype=:line, label="predict", leg=true)
# savefig(p1, "var1.svg")
# plot!(X_train[:,1][x_perm_1], pred_train_linear[x_perm_1], color = "red", mswidth=0, msize=3, label = "Linear", st=:scatter, leg=false)

p2 = plot(X_train[:,2], Y_train, ms = 3, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("target"), leg = false, cbar=true)
p2_bin = plot(X_bin[:,2], Y_train, ms = 3, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("target"), leg = false, cbar=true, label="")
plot!(fill(μ, params1.nbins), lw=3, color="#66ffcc", background_color = "white", seriestype=:line, label="predict", leg = true)
# savefig(p2, "var2.svg")
# plot!(X_train[:,2][x_perm_2], pred_train_linear[x_perm_2], color = "red", mswidth=0, msize=3, st=:scatter, label = "Predict")

p = plot(p1,p2, layout=(2,1))
savefig(p, "blog/raw_one_ways.svg")
savefig(p, "blog/raw_one_ways.png")
p = plot(p1_bin, p2_bin, layout=(2,1))
savefig(p, "blog/bin_one_ways.svg")
savefig(p, "blog/bin_one_ways.png")

# train iteration
# plot left vs right points
left_id = X_bin[:,2] .== 1
p = plot(X_bin[left_id,2], Y_train[left_id], ms = 3, color="darkred", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("residual"), leg = true, cbar=true, label="left")
plot!(X_bin[.!left_id,2], Y_train[.!left_id], ms = 3, color="#33ccff", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("residual"), leg = true, cbar=true, label="right")
plot!(fill(μ, params1.nbins), lw=3, color="#66ffcc", background_color = "white", seriestype=:line, label="predict", leg = true)

# residuals
residuals = Y_train .- μ
left_id = X_bin[:,2] .== 1
p = plot(X_bin[left_id,2], residuals[left_id], ms = 3, color="darkred", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("residual"), leg = true, cbar=true, label="left")
plot!(X_bin[.!left_id,2], residuals[.!left_id], ms = 3, color="#33ccff", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("var2"), yaxis = ("residual"), leg = true, cbar=true, label="right")
plot!(fill(0, params1.nbins), lw=3, color="#66ffcc", background_color = "white", seriestype=:line, label="", leg = true)
savefig(p, "blog/first_split.svg")
savefig(p, "blog/first_split.png")

left_res = residuals[left_id]
function loss(x::Vector, val)
    sum((x .- val).^2)
end
eval_pts = -2.1:0.01:0.5
left_loss = loss.(Ref(left_res), eval_pts)
p = plot(left_res, left_res .* 0, ms = 5, color="darkred", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("predict"), yaxis = ("loss"), leg = true, cbar=true, label="observation")
# plot!(left_res, left_loss, ms = 3, color="#33ccff", msw=0, background_color = "white", seriestype=:scatter, xaxis = ("residual"), yaxis = ("loss"), leg = true, cbar=true, label="")
plot!(eval_pts, left_loss, lw=3, color="#66ffcc", background_color = "white", seriestype=:line, label="loss", leg = true)
savefig(p, "blog/left_parabole.svg")
savefig(p, "blog/left_parabole.png")
loss(left_res, 0.0) - loss(left_res, mean(left_res))

# add single pt parabol
left_res_1 = sort(left_res)[35]
left_loss_1 = loss.(Ref([left_res_1]), eval_pts)
plot!(eval_pts, left_loss_1, lw=0.5, color="#33ccff", background_color = "white", seriestype=:line, label="", leg = true)

###########################
# raw compute
###########################
𝑖 = collect(1:size(X_train,1))
δ, δ² = zeros(SVector{model.K, Float64}, size(X_train, 1)), zeros(SVector{model.K, Float64}, size(X_train, 1))
𝑤 = zeros(SVector{1, Float64}, size(X_train, 1)) .+ 1
pred = zeros(SVector{model.K,Float64}, size(X_train,1)) .+= μ
EvoTrees.update_grads!(params1.loss, params1.α, pred, Y_train, δ, δ², 𝑤)
∑δ, ∑δ², ∑𝑤 = sum(δ[left_id]), sum(δ²[left_id]), sum(𝑤[left_id])
gain = EvoTrees.get_gain(params1.loss, ∑δ, ∑δ², ∑𝑤, params1.λ)

#####################################
# 3D visualisation of data
#####################################
# p = plot(X_train[:,1], X_train[:,2], Y_train, zcolor=Y_train, color=cgrad(["red","#3399ff"]), msize=5, markerstrokewidth=0, leg=false, cbar=true, w=1, st=:scatter)
p = plot(X_grid[:,1], X_grid[:,2], Y_grid, zcolor=Y_grid, color=cgrad(["#555555", "#eeeeee"]), msize=5, markerstrokewidth=0, leg=false, cbar=true, st=:scatter, xaxis="var1", yaxis="var2")
plot!(X_train[:,1], X_train[:,2], Y_train, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msize=4, markerstrokewidth=0, st=:scatter)
savefig(p, "blog/data_3D.svg")
savefig(p, "blog/data_3D.png")
plot(X_train[:,1], X_train[:,2], pred_train_linear, zcolor=Y_train, m=(5, 0.9, :rainbow, Plots.stroke(0)), leg=false, cbar=true, w=1, st=:scatter)
plot(X_train[:,1], X_train[:,2], pred_train_linear, zcolor=Y_train, st=[:surface], leg=false, cbar=true, fillcolor=:rainbow, markeralpha=1.0)

p_bin = plot(X_bin[:,1], X_bin[:,2], Y_train, zcolor=Y_train, color=cgrad(["darkred", "#33ccff"]), msize=4, markerstrokewidth=0, st=:scatter, leg=false, cbar=true)

gr()

params1 = EvoTreeRegressor(
    loss=:linear, metric=:mse,
    nrounds=100, nbins = 100,
    λ = 0.0, γ=0.0, η=0.5,
    max_depth = 2, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

anim = @animate for i=1:20
    params1.nrounds = (i-1)*5+1
    model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = Inf)
    pred_train_linear = predict(model, X_train)
    x_perm = sortperm(X_train[:,1])
    plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
    plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
end
gif(anim, "blog/anim_fps1.gif", fps = 1)

# tree vec
function treevec(tree)
    source, target = zeros(Int, max(1, length(tree.nodes)-1)), zeros(Int, max(1, length(tree.nodes)-1))
    count_s, count_t = 1, 1
    for i in 1:length(tree.nodes)
        if tree.nodes[i].split
            source[count_s] = i
            source[count_s+1] = i
            target[count_t] = tree.nodes[i].left
            target[count_t+1] = tree.nodes[i].right
            count_s += 2
            count_t += 2
        elseif i ==1
            source[i] = i
            target[i] = i
        end
    end
    return source, target
end

# plot tree
function nodenames(tree)
    names = []
    for i in 1:length(tree.nodes)
        if tree.nodes[i].split
            push!(names, "feat: " * string(tree.nodes[i].feat) * "\n< " * string(round(tree.nodes[i].cond, sigdigits=3)))
        else
            push!(names, "pred:\n" * string(round(tree.nodes[i].pred[1], sigdigits=3)))
        end
    end
    return names
end

tree1 = model.trees[2]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
# p1 = graphplot(source, target, method=:tree, names = nodes, linecolor=:brown, nodeshape=:hexagon, fontsize=8, fillcolor="#66ffcc")
p1 = graphplot(source, target, method=:buchheim, names = nodes, linecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc")

tree1 = model.trees[3]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p2 = graphplot(source, target, method=:buchheim, names = nodes, linecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc")

tree1 = model.trees[50]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p3 = graphplot(source, target, method=:buchheim, names = nodes, linecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc")

tree1 = model.trees[90]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p4 = graphplot(source, target, method=:buchheim, names = nodes, edgecolor=:black, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc")
default(size=(1600, 1600))
seed!(1)
fills = sample(["#33ffcc", "#99ccff"], length(source), replace=true)
fills = sample(["lightgray", "#33ffcc"], length(source), replace=true)
p4 = graphplot(source, target, method=:buchheim, root=:top, names = nodes, edgecolor=:black, nodeshape=:hexagon, fontsize=9, axis_buffer=0.05, nodesize=0.025, nodecolor=fills)


p = plot(p1,p2,p3,p4)
savefig(p, "blog/tree_group.svg")
savefig(p1, "blog/tree_1.svg")
savefig(p, "blog/tree_group.png")
savefig(p1, "blog/tree_1.png")




################################
# animation on one way plot
################################
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
    nrounds=10, nbins = 32,
    λ = 0.0, γ=0.0, η=0.5,
    max_depth = 3, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)

anim = @animate for i=0:4
    params1.nrounds = i
    # i == 1 ? params1.η = 0.001 : params1.η = 0.5
    model = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = Inf)
    pred_train_linear = predict(model, X_train)
    x_perm = sortperm(X_train[:,1])
    plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
    plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
end
gif(anim, "blog/anim_fps1.gif", fps = 1)

ps = [p1,p2,p3,p4]
anim = @animate for i=0:4
    # i == 1 ? params1.η = 0.001 : params1.η = 0.5
    # if i == 1
    #     p = plot(p1)
    # elseif i ==2
    #     p = plot(p1,p2)
    # end
    if i==0
        plot(foreground_color_subplot=:white)
    else
        plot(ps[1:i]...)
    end
end
gif(anim, "blog/anim_tree.gif", fps = 1)


tree1 = model.trees[2]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p1 = graphplot(source, target, method=:tree, names = nodes, edgecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc", nodestrokewidth=0)

tree1 = model.trees[3]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p2 = graphplot(source, target, method=:tree, names = nodes, edgecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc", nodestrokewidth=0)

tree1 = model.trees[4]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p3 = graphplot(source, target, method=:tree, names = nodes, edgecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc", nodestrokewidth=0)

tree1 = model.trees[5]
source, target = treevec(tree1)
nodes = nodenames(tree1)
seed!(1)
p4 = graphplot(source, target, method=:tree, names = nodes, edgecolor=:brown, nodeshape=:hexagon, fontsize=8, nodecolor="#66ffcc", nodestrokewidth=0)

p = plot(p1,p2,p3,p4)
savefig(p, "blog/tree_group.svg")
savefig(p1, "blog/tree_1.svg")
savefig(p, "blog/tree_group.png")
savefig(p1, "blog/tree_1.png")
