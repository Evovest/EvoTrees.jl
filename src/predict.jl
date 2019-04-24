# prediction from single tree - assign each observation to its final leaf
function predict!(pred, tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    @threads for i in 1:size(X, 1)
        id = 1
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
function predict(tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(size(X, 1))
    predict!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(size(X, 1))
    for tree in model.trees
        predict!(pred, tree, X)
    end
    if model.params.loss == :logistic
        @. pred = sigmoid(pred)
    elseif model.params.loss == :poisson
        @. pred = exp(pred)
    end
    return pred
end
