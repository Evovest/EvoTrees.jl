# prediction from single tree - assign each observation to its final leaf
function predict(tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(size(X, 1))
    @threads for i in 1:size(X, 1)
        id = Int(1)
        x = view(X, i, :)
        while tree.nodes[id].split
            if x[tree.nodes[id].feat] <= tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
    return pred
end

# prediction from single tree - assign each observation to its final leaf
# function predict!(pred, tree::Tree, X)
#
#     @threads for i in 1:size(X, 1)
#         # for i in 1:size(X, 1)
#         node = tree.nodes[1]
#         x = view(X, i, :)
#         # x = X[i, :]
#         while isa(node, SplitNode)
#             id = node.feat
#             cond = node.cond
#             if x[id] <= cond
#                 node = tree.nodes[node.left]
#             else
#                 node = tree.nodes[node.right]
#             end
#         end
#         pred[i] += node.pred
#     end
#     return pred
# end

# prediction from single tree - assign each observation to its final leaf
function predict!(pred, tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    @threads for i in 1:size(X, 1)
        id = Int(1)
        x = view(X, i, :)
        while tree.nodes[id].split
            if x[tree.nodes[id].feat] <= tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
    return pred
end



# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTree, X)
    pred = zeros(size(X, 1))
    @threads for i in 1:size(X, 1)
        x = view(X, i, :)
        for tree in model.trees
            id = Int(1)
            while tree.nodes[id].split
                if x[tree.nodes[id].feat] <= tree.nodes[id].cond
                    id = tree.nodes[id].left
                else
                    id = tree.nodes[id].right
                end
            end
            pred[i] += tree.nodes[id].pred
        end
    end
    return pred
end
