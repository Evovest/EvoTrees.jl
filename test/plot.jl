using Random
using Plots
using Revise
using EvoTrees

# @load "blog/model_linear.bson" model
# @load "data/model_linear_8.bson" model
# @load "data/model_gaussian_5.bson" model

model = EvoTrees.load("data/model_linear_4.bson");
var_names = ["var_$i" for i in 1:100]
plot(model)
plot(model, 2)
plot(model, 3, var_names)
plot(model.trees[2])
plot(model.trees[2], var_names)

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
