
# prediction from single tree - assign each observation to its final leaf
function predict(tree::TreeNode, X)

    # pred = Vector{AbstractFloat}(undef, size(X, 1))
    pred = zeros(size(X, 1))

    @threads for i in 1:size(X, 1)
    # for i in 1:size(X, 1)
        node = tree
        # x = view(X, i, :)
        x = X[i, :]
        while isa(node, TreeSplit)
            id = node.feat
            cond = node.cond
            if x[id] <= cond
                node = node.left
            else
                node = node.right
            end
        end
        # pred[i] = node.pred
        pred[i] += node.pred
    end
    return pred
end


function predict(model::GBTree, X)

    pred = zeros(size(X, 1))

    @threads for i in 1:size(X, 1)
        # for i in 1:size(X, 1)
        # x = view(X, i, :)
        x = X[i, :]
        for tree in model.trees
            node = tree
            while isa(node, TreeSplit)
                id = node.feat
                cond = node.cond
                if x[id] <= cond
                    node = node.left
                else
                    node = node.right
                end
            end
            pred[i] += node.pred
        end
    end
    return pred
end



# prediction from single tree - assign each observation to its final leaf
function predict(tree::Tree, X)

    # pred = Vector{AbstractFloat}(undef, size(X, 1))
    pred = zeros(size(X, 1))

    @threads for i in 1:size(X, 1)
    # for i in 1:size(X, 1)
        node = tree.nodes[1]
        x = view(X, i, :)
        # x = X[i, :]
        while node.feat > 0
            id = node.feat
            cond = node.cond
            if x[id] <= cond
                node = tree.nodes[node.left]
            else
                node = tree.nodes[node.right]
            end
        end
        # pred[i] = node.pred
        pred[i] += node.pred
    end
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict!(pred, tree::Tree, X)

    @threads for i in 1:size(X, 1)
    # for i in 1:size(X, 1)
        node = tree.nodes[1]
        x = view(X, i, :)
        # x = X[i, :]
        while node.feat > 0
            id = node.feat
            cond = node.cond
            if x[id] <= cond
                node = tree.nodes[node.left]
            else
                node = tree.nodes[node.right]
            end
        end
        # pred[i] = node.pred
        pred[i] += node.pred
    end
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTrees, X)

    # pred = Vector{AbstractFloat}(undef, size(X, 1))
    pred = zeros(size(X, 1))

    @threads for i in 1:size(X, 1)
        # for i in 1:size(X, 1)
        # x = view(X, i, :)
        x = X[i, :]
        for tree in model.trees
            node = tree.nodes[1]
            while node.feat > 0
                id = node.feat
                cond = node.cond
                if x[id] <= cond
                    node = tree.nodes[node.left]
                else
                    node = tree.nodes[node.right]
                end
            end
            # pred[i] = node.pred
            pred[i] += node.pred
        end
    end
    return pred
end
