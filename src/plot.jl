function get_adj_list(tree::EvoTrees.Tree)
    adj = Vector{Vector{Int}}()
    for node in tree.nodes
        node.split ? push!(adj, [node.left, node.right]) : push!(adj, [])
    end
    return adj
end

function get_shapes(tree_layout)
    shapes = Vector(undef, length(tree_layout))
    for i in 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        x_buff = 0.45
        y_buff = 0.45
        shapes[i] = [(x - x_buff, y + y_buff), (x + x_buff, y + y_buff), (x + x_buff, y - y_buff), (x - x_buff, y - y_buff)]
    end
    return shapes
end

function get_annotations(tree_layout, tree, var_names)
    # annotations = Vector{Tuple{Float64, Float64, String, Tuple}}(undef, length(tree_layout))
    annotations = []
    for i in 1:length(tree_layout)
        x, y = tree_layout[i][1], tree_layout[i][2] # center point
        if tree.nodes[i].split
            feat = isnothing(var_names) ? "feat: " *  string(tree.nodes[i].feat) : var_names[tree.nodes[i].feat]
            txt = "$feat\n" * string(round(tree.nodes[i].cond, sigdigits=3))
        else
            txt = "pred:\n" *  string(round(tree.nodes[i].pred[1], sigdigits=3))
        end
        # annotations[i] = (x, y, txt, (9, :white, "helvetica"))
        push!(annotations, (x, y, txt, 10))
    end
    return annotations
end

function get_curves(adj, tree_layout, shapes)
    curves = []
    num_curves = sum(length.(adj))
    for i in 1:length(adj)
        for j in 1:length(adj[i])
            # curves is a length 2 tuple: (vector Xs, vector Ys)
            push!(curves, ([tree_layout[i][1], tree_layout[adj[i][j]][1]], [shapes[i][3][2], shapes[adj[i][j]][1][2]]))
        end
    end
    return curves
end

@recipe function plot(tree::EvoTrees.Tree, var_names=nothing)

    adj = EvoTrees.get_adj_list(tree)
    tree_layout = length(adj) == 1 ? [[0.0,0.0]] : NetworkLayout.Buchheim.layout(adj)
    shapes = EvoTrees.get_shapes(tree_layout) # issue with Shape coming from Plots... to be converted o Shape in Receipe?
    annotations = EvoTrees.get_annotations(tree_layout, tree, var_names) # same with Plots.text
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
    
    for i in 1:length(shapes)
        @series begin
            fillcolor = length(adj[i]) == 0 ? "#84DCC6" : "#C8D3D5"
            fillcolor --> fillcolor
            seriestype --> :shape
            return shapes[i] 
        end
    end

    for i in 1:length(curves)
        @series begin
            seriestype --> :curves
            return curves[i] 
        end
    end   
end

@recipe function plot(model::EvoTrees.GBTree, n=1, var_names=nothing)

    isnothing(var_names)

    tree = model.trees[n]
    adj = EvoTrees.get_adj_list(tree)
    tree_layout = length(adj) == 1 ? [[0.0,0.0]] : NetworkLayout.Buchheim.layout(adj)
    shapes = EvoTrees.get_shapes(tree_layout) # issue with Shape coming from Plots... to be converted o Shape in Receipe?
    annotations = EvoTrees.get_annotations(tree_layout, tree, var_names) # same with Plots.text
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
    
    for i in 1:length(shapes)
        @series begin
            fillcolor = length(adj[i]) == 0 ? "#84DCC6" : "#C8D3D5"
            fillcolor --> fillcolor
            seriestype --> :shape
            return shapes[i] 
        end
    end

    for i in 1:length(curves)
        @series begin
            seriestype --> :curves
            return curves[i] 
        end
    end
end