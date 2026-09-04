using EvoTrees: Tree, MSE, EvoTree, EvoTreeRegressor, fit

function _split_tree(; K=1, pred_left=1.5f0, pred_right=-0.5f0)
    tree = Tree{MSE,K}(2)
    tree.split[1] = true
    tree.feat[1] = 1
    tree.cond_bin[1] = 0x01
    tree.pred[1, 2] = pred_left
    tree.pred[1, 3] = pred_right
    K > 1 && (tree.pred[2, 2] = 0.25f0; tree.pred[2, 3] = 0.75f0)
    return tree
end

function _model_from_trees(tree; feature_names=[:feat_1], edges=[[0.5, 1.0, 1.5]], feattypes=[true], K=1)
    info = Dict{Symbol,Any}(:feature_names => feature_names, :edges => edges, :feattypes => feattypes)
    return EvoTree{MSE,K}(MSE, K, zeros(K), [tree], info)
end

@testset "tree plot spec" begin
    tree = _split_tree()
    ids, children = EvoTrees.tree_children(tree)
    @test ids == [1, 2, 3]
    @test children == [[2, 3], Int[], Int[]]
    xs, ys = EvoTrees.layout_tree(children)
    @test ys[1] > ys[2] == ys[3]
    @test xs[2] < xs[1] < xs[3]

    spec = EvoTrees.tree_plot_spec(_model_from_trees(tree), 1)
    @test spec.isleaf == [false, true, true]
    @test occursin("feat_1", spec.labels[1]) && occursin("≤", spec.labels[1]) && occursin("0.5", spec.labels[1])
    @test spec.labels[2] == "1.5" && spec.labels[3] == "-0.5"

    cat_spec = EvoTrees.tree_plot_spec(_model_from_trees(tree; feature_names=[:color], edges=[["red", "blue"]], feattypes=[false]), 1)
    @test occursin("=", cat_spec.labels[1]) && occursin("red", cat_spec.labels[1])

    multi_spec = EvoTrees.tree_plot_spec(_model_from_trees(_split_tree(; K=2); K=2), 1)
    @test occursin("1.5", multi_spec.labels[2]) && occursin("0.25", multi_spec.labels[2])
end

@testset "tree plot from fit" begin
    x = reshape(Float32.(1:32), :, 1)
    y = Float32.(x[:, 1] .> 16)
    model = fit(EvoTreeRegressor(; loss=:mse, nrounds=2, max_depth=3, nbins=8, eta=1.0, seed=1); x_train=x, y_train=y)
    spec = EvoTrees.tree_plot_spec(model, 1)
    @test length(spec.xs) == length(spec.labels) == length(spec.isleaf)
    @test any(spec.isleaf) && all(!isempty, spec.labels)
end
