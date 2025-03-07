function get_adj_list(tree::EvoTrees.Tree)
    n = 1
    map = ones(Int, 1)
    adj = Vector{Vector{Int}}()

    if tree.split[1]
        push!(adj, [n + 1, n + 2])
        n += 2
    else
        push!(adj, [])
    end

    for i = 2:length(tree.split)
        if tree.split[i]
            push!(map, i)
            push!(adj, [n + 1, n + 2])
            n += 2
        elseif tree.split[i>>1]
            push!(map, i)
            push!(adj, [])
        end
    end
    return (map=map, adj=adj)
end

function get_shapes(tree_layout)
    shapes = Vector(undef, length(tree_layout))
    for i = eachindex(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        x_buff = 0.45
        y_buff = 0.45
        shapes[i] = [
            (x - x_buff, y + y_buff),
            (x + x_buff, y + y_buff),
            (x + x_buff, y - y_buff),
            (x - x_buff, y - y_buff),
        ]
    end
    return shapes
end

function get_annotations(tree_layout, map, tree, feature_names, edges)
    # annotations = Vector{Tuple{Float64, Float64, String, Tuple}}(undef, length(tree_layout))
    annotations = []
    for i = eachindex(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        if tree.split[map[i]]
            fidx = tree.feat[map[i]]
            fname = feature_names[fidx]
            bin = tree.cond_bin[map[i]]
            value = edges[fidx][bin]
            typeof(value) <: Number ? value = round(value, sigdigits=3) : nothing
            txt = "$fname\n$value"
        else
            txt = string(round(tree.pred[1, map[i]], sigdigits=3))
        end
        # annotations[i] = (x, y, txt, (9, :white, "helvetica"))
        push!(annotations, (x, y, txt, 10))
    end
    return annotations
end

function get_curves(adj, tree_layout, shapes)
    curves = []
    num_curves = sum(length.(adj))
    for i = eachindex(adj)
        for j = eachindex(adj[i])
            # curves is a length 2 tuple: (vector Xs, vector Ys)
            push!(
                curves,
                (
                    [tree_layout[i][1], tree_layout[adj[i][j]][1]],
                    [shapes[i][3][2], shapes[adj[i][j]][1][2]],
                ),
            )
        end
    end
    return curves
end

@recipe function plot(model::EvoTrees.EvoTree, n=2)

    feature_names = model.info[:feature_names]
    edges = model.info[:edges]
    tree = model.trees[n]
    map, adj = EvoTrees.get_adj_list(tree)
    tree_layout = length(adj) == 1 ? [[0.0, 0.0]] : NetworkLayout.buchheim(adj)
    shapes = EvoTrees.get_shapes(tree_layout) # issue with Shape coming from Plots... to be converted o Shape in Receipe?
    annotations = EvoTrees.get_annotations(tree_layout, map, tree, feature_names, edges) # same with Plots.text
    curves = EvoTrees.get_curves(adj, tree_layout, shapes)

    size_base = floor(log2(length(adj)))
    size = (128 * 2^size_base, 96 * (1 + size_base))

    background_color --> :white
    linecolor --> :black
    legend --> nothing
    axis --> nothing
    framestyle --> :none
    size --> size
    annotations --> annotations

    for i = eachindex(shapes)
        @series begin
            _color = length(adj[i]) == 0 ? "#26a671" : "#e6ebf1"
            fillcolor --> _color
            linewidth --> 0
            seriestype --> :shape
            return shapes[i]
        end
    end

    for i = eachindex(curves)
        @series begin
            seriestype --> :curves
            return curves[i]
        end
    end
end