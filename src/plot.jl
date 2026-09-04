"""
    treeplot(model::EvoTree, n=1)
    treeplot!(ax, model::EvoTree, n=1)

Plot tree `n` of a fitted `EvoTree`. Default `n = 1` is the first boosting tree.
Requires a Makie backend (`CairoMakie` or `GLMakie`). `plot(model, n)` works once a backend is loaded.
"""
function treeplot end
function treeplot! end

struct TreePlotSpec
    xs::Vector{Float64}
    ys::Vector{Float64}
    isleaf::Vector{Bool}
    labels::Vector{String}
    edges::Vector{Tuple{Int,Int}}
end

function tree_children(tree::Tree)
    ids = Int[]
    function visit!(i)
        push!(ids, i)
        if tree.split[i]
            visit!(2i)
            visit!(2i + 1)
        end
    end
    visit!(1)
    id_to_layout = Dict(id => k for (k, id) in enumerate(ids))
    children = [Int[] for _ in ids]
    for (k, id) in enumerate(ids)
        tree.split[id] && (children[k] = [id_to_layout[2id], id_to_layout[2id + 1]])
    end
    return ids, children
end

function layout_tree(children; xgap=2.0, ygap=1.6)
    n = length(children)
    xs = zeros(n)
    depths = zeros(Int, n)
    function set_depth!(i, d)
        depths[i] = d
        for c in children[i]
            set_depth!(c, d + 1)
        end
    end
    set_depth!(1, 0)
    next_x = 0.0
    function place!(i)
        ch = children[i]
        if isempty(ch)
            xs[i] = next_x
            next_x += xgap
        else
            foreach(place!, ch)
            xs[i] = 0.5 * (xs[first(ch)] + xs[last(ch)])
        end
    end
    place!(1)
    maxd = maximum(depths)
    return xs, [(maxd - d) * ygap for d in depths]
end

_fmt(x::Number) = string(round(Float64(x); sigdigits=3))
_fmt(x) = string(x)

function tree_plot_spec(model::EvoTree, n::Integer=1)
    tree = model.trees[n]
    fnames, edges, feattypes = model.info[:feature_names], model.info[:edges], model.info[:feattypes]
    ids, children = tree_children(tree)
    xs, ys = layout_tree(children)
    labels = map(ids) do i
        if tree.split[i]
            f = tree.feat[i]
            # numeric: x_bin ≤ cond_bin ⇔ x ≤ edges[cond_bin] (searchsortedfirst)
            op = feattypes[f] ? "≤" : "="
            "$(fnames[f])\n$op $(_fmt(edges[f][tree.cond_bin[i]]))"
        else
            p = view(tree.pred, :, i)
            join(_fmt.(p), length(p) <= 2 ? "\n" : ", ")
        end
    end
    elist = Tuple{Int,Int}[(i, c) for (i, ch) in enumerate(children) for c in ch]
    return TreePlotSpec(xs, ys, [isempty(ch) for ch in children], labels, elist)
end

function treeplot_size(spec::TreePlotSpec)
    d = max(0, floor(Int, log2(max(1, length(spec.xs)))))
    return (max(256, 128 * 2^d), max(200, 96 * (1 + d)))
end
