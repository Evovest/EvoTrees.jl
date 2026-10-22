module EvoTreesMakieExt

using EvoTrees
using Makie
import EvoTrees: treeplot, treeplot!

@recipe TreePlot (spec,) begin
    "Fill color of split (internal) nodes."
    nodecolor = "#e6ebf1"
    "Fill color of leaf nodes."
    leafcolor = "#26a671"
    "Color of node labels."
    textcolor = :black
    "Font size of node labels."
    fontsize = 11
    "Width of node boxes in data coordinates."
    boxwidth = 1.55
    "Height of node boxes in data coordinates."
    boxheight = 0.85
    "Color of edges connecting parent and child nodes."
    linecolor = :black
    "Width of edges connecting parent and child nodes."
    linewidth = 1.0
    "Stroke color of node boxes."
    strokecolor = "#c5ccd6"
    "Stroke width of node boxes."
    strokewidth = 0.6
    Makie.mixin_generic_plot_attributes()...
end

Makie.convert_arguments(::Type{<:TreePlot}, spec::EvoTrees.TreePlotSpec) = (spec,)
Makie.convert_arguments(::Type{<:TreePlot}, model::EvoTrees.EvoTree, n::Integer) =
    (EvoTrees.tree_plot_spec(model, Int(n)),)
Makie.convert_arguments(::Type{<:TreePlot}, model::EvoTrees.EvoTree) =
    (EvoTrees.tree_plot_spec(model),)

Makie.plottype(::EvoTrees.EvoTree) = TreePlot
Makie.plottype(::EvoTrees.EvoTree, ::Integer) = TreePlot
Makie.plottype(::EvoTrees.TreePlotSpec) = TreePlot
Makie.preferred_axis_type(::TreePlot) = Axis
function Makie.preferred_axis_attributes(::Type{Axis}, ::TreePlot)
    return (;
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        xgridvisible=false, ygridvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false,
    )
end

function Makie.plot!(p::TreePlot)
    spec = to_value(p[1])
    w, h = to_value(p.boxwidth), to_value(p.boxheight)
    split_rects = Rect2f[]
    leaf_rects = Rect2f[]
    for i in eachindex(spec.xs)
        rect = Rect2f(spec.xs[i] - w / 2, spec.ys[i] - h / 2, w, h)
        push!(spec.isleaf[i] ? leaf_rects : split_rects, rect)
    end
    segs = Point2f[]
    for (a, b) in spec.edges
        push!(segs, Point2f(spec.xs[a], spec.ys[a] - h / 2), Point2f(spec.xs[b], spec.ys[b] + h / 2))
    end
    strokecolor, strokewidth = to_value(p.strokecolor), to_value(p.strokewidth)
    isempty(split_rects) || poly!(p, split_rects; color=to_value(p.nodecolor), strokecolor, strokewidth)
    isempty(leaf_rects) || poly!(p, leaf_rects; color=to_value(p.leafcolor), strokecolor, strokewidth)
    isempty(segs) || linesegments!(p, segs; color=to_value(p.linecolor), linewidth=to_value(p.linewidth))
    text!(p, Point2f.(spec.xs, spec.ys); text=spec.labels, align=(:center, :center),
        justification=:center, color=to_value(p.textcolor), fontsize=to_value(p.fontsize))
    return p
end

function treeplot(model::EvoTrees.EvoTree, n::Integer=1; figure=NamedTuple(), kwargs...)
    spec = EvoTrees.tree_plot_spec(model, n)
    return plot(spec; figure=merge((; size=EvoTrees.treeplot_size(spec)), figure), kwargs...)
end

end
