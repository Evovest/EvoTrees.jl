using Random
using RecipesBase
using NetworkLayout
using BSON: @save, @load
using Plots
using Revise
using EvoTrees

# @load "blog/model_linear.bson" model
@load "model.bson" model

# tree vec
function get_adj_list(tree::EvoTrees.Tree)
    adj = Vector{Vector{Int}}()
    for node in tree.nodes
        node.split ? push!(adj, [node.left, node.right]) : push!(adj, [])
    end
    return adj
end

function get_shapes(tree_layout)
    shapes = Vector{Shape}(undef, length(tree_layout))
    for i in 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        x_buff = 0.45
        y_buff = 0.45
        shapes[i] = Shape([(x - x_buff, y + y_buff), (x + x_buff, y + y_buff), (x + x_buff, y - y_buff), (x - x_buff, y - y_buff)])
    end
    return shapes
end

function get_annotations(tree_layout, tree)
    annotations = Vector{Tuple{Float64,Float64,Plots.PlotText}}(undef, length(tree_layout))
    for i in 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        if tree.nodes[i].split
            txt = "feat: " *  string(tree.nodes[i].feat) * "\n" * string(round(tree.nodes[i].cond, sigdigits=3))
        else
            txt = "pred:\n" *  string(round(tree.nodes[i].pred[1], sigdigits=3))
        end
        annotations[i] = (x, y, Plots.text(txt, 9, :white, :center, "helvetica"))
    end
    return annotations
end

function get_curves(tree_layout, tree)
    annotations = Vector{Tuple{Float64,Float64,Plots.PlotText}}(undef, length(tree_layout))
    for i in 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        if tree.nodes[i].split
            txt = "feat: " *  string(tree.nodes[i].feat) * "\n" * string(round(tree.nodes[i].cond, sigdigits=3))
        else
            txt = "pred:\n" *  string(round(tree.nodes[i].pred[1], sigdigits=3))
        end
        annotations[i] = (x, y, Plots.text(txt, 9, :white, :center, "helvetica"))
    end
    return annotations
end

tree = model.trees[30]
adj = get_adj_list(tree)
tree_layout = NetworkLayout.Buchheim.layout(adj)
shapes = get_shapes(tree_layout)
annotations = get_annotations(tree_layout, tree)

plot(shapes, bg=:white, fc="#023572", lc=:gray, legend=nothing, axis=nothing, framestyle=:none, size = (1200, 400))
# plot!(shapes, bg=:white, fc="#023572", lc=:gray, legend=nothing, axis=nothing, framestyle=:none)
annotate!(annotations)

curve_1 = [tree_layout[1][1], tree_layout[2][1]], [tree_layout[1][2], tree_layout[2][2]]
curve_2 = [tree_layout[1][1], tree_layout[3][1]], [tree_layout[1][2], tree_layout[3][2]]
curves!([curve_1, curve_2], color=:black)
plot([curve_1, curve_2], bg=:white, fc="#023572", lc=:gray, legend=nothing, axis=nothing, framestyle=:none)

typeof(tree_layout[1])
BezierCurve(tree_layout[1])


mutable struct BCurve{T <: GeometryBasics.Point}
    control_points::Vector{T}
end

function (bc::BCurve)(t::Real)
    p = zero(P2)
    n = length(bc.control_points) - 1
    for i in 0:n
        p += bc.control_points[i + 1] * binomial(n, i) * (1 - t)^(n - i) * t^i
    end
    p
end
