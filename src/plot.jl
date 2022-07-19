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
    for i = 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        x_buff = 0.45
        y_buff = 0.45
        shapes[i] = [(x - x_buff, y + y_buff), (x + x_buff, y + y_buff), (x + x_buff, y - y_buff), (x - x_buff, y - y_buff)]
    end
    return shapes
end

function get_annotations(tree_layout, map, tree, var_names)
    # annotations = Vector{Tuple{Float64, Float64, String, Tuple}}(undef, length(tree_layout))
    annotations = []
    for i = 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        if tree.split[map[i]]
            feat = isnothing(var_names) ? "feat: " * string(tree.feat[map[i]]) : var_names[tree.feat[map[i]]]
            txt = "$feat\n" * string(round(tree.cond_float[map[i]], sigdigits=3))
        else
            txt = "pred:\n" * string(round(tree.pred[1, map[i]], sigdigits=3))
        end
        # annotations[i] = (x, y, txt, (9, :white, "helvetica"))
        push!(annotations, (x, y, txt, 10))
    end
    return annotations
end

function get_curves(adj, tree_layout, shapes)
    curves = []
    num_curves = sum(length.(adj))
    for i = 1:length(adj)
        for j = 1:length(adj[i])
            # curves is a length 2 tuple: (vector Xs, vector Ys)
            push!(curves, ([tree_layout[i][1], tree_layout[adj[i][j]][1]], [shapes[i][3][2], shapes[adj[i][j]][1][2]]))
        end
    end
    return curves
end

@recipe function plot(tree::EvoTrees.Tree, var_names=nothing)

    map, adj = EvoTrees.get_adj_list(tree)
    tree_layout = length(adj) == 1 ? [[0.0, 0.0]] : NetworkLayout.buchheim(adj)
    shapes = EvoTrees.get_shapes(tree_layout) # issue with Shape coming from Plots... to be converted o Shape in Receipe?
    annotations = EvoTrees.get_annotations(tree_layout, map, tree, var_names) # same with Plots.text
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

    for i = 1:length(shapes)
        @series begin
            fillcolor = length(adj[i]) == 0 ? "#84DCC6" : "#C8D3D5"
            fillcolor --> fillcolor
            seriestype --> :shape
            return shapes[i]
        end
    end

    for i = 1:length(curves)
        @series begin
            seriestype --> :curves
            return curves[i]
        end
    end
end

@recipe function plot(model::EvoTrees.GBTree, n=1, var_names=nothing)

    isnothing(var_names)

    tree = model.trees[n]
    map, adj = EvoTrees.get_adj_list(tree)
    tree_layout = length(adj) == 1 ? [[0.0, 0.0]] : NetworkLayout.buchheim(adj)
    shapes = EvoTrees.get_shapes(tree_layout) # issue with Shape coming from Plots... to be converted o Shape in Receipe?
    annotations = EvoTrees.get_annotations(tree_layout, map, tree, var_names) # same with Plots.text
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

    for i = 1:length(shapes)
        @series begin
            fillcolor = length(adj[i]) == 0 ? "#84DCC6" : "#C8D3D5"
            fillcolor --> fillcolor
            seriestype --> :shape
            return shapes[i]
        end
    end

    for i = 1:length(curves)
        @series begin
            seriestype --> :curves
            return curves[i]
        end
    end
end