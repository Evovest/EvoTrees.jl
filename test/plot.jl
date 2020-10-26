using Random
using RecipesBase
using NetworkLayout
using BSON: @save, @load
using Plots
using Revise
using EvoTrees

@load "blog/model_linear.bson" model

# tree vec
function get_adj_list(tree::EvoTrees.Tree)
    adj = Vector{Vector{Int}}()
    for node in tree.nodes
        node.split ? push!(adj, [node.left, node.right]) : push!(adj, [])
    end
    return adj
end

adj = get_adj_list(model.trees[2])
tree_layout = NetworkLayout.Buchheim.layout(adj)

s1 = Shape([0.5, 1, 1, 0.5], [0.5, 0.5, 0, 0])
s2 = deepcopy(s1)
s2.x .-= 1.5
s_tot = [s1, s2]

scatter(tree_layout, bg=:white, shape=:rect, mc=:lightblue, msc=:white, legend=nothing, axis=nothing, framestyle=:none)
plot!(s_tot, fc=:lightgray, lc=nothing)

typeof(tree_layout[1])
BezierCurve(tree_layout[1])


mutable struct BCurve{T <: GeometryBasics.Point}
    control_points::Vector{T}
end

function (bc::BCurve)(t::Real)
    p = zero(P2)
    n = length(bc.control_points)-1
    for i in 0:n
        p += bc.control_points[i+1] * binomial(n, i) * (1-t)^(n-i) * t^i
    end
    p
end
