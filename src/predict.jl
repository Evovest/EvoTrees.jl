# prediction from single tree - assign each observation to its final leaf
function predict!(pred, tree::Tree, X::AbstractArray{T, 2}) where T<:Real
    @threads for i in 1:size(X, 1)
        id = 1
        x = view(X, i, :)
        while tree.nodes[id].split
            if x[tree.nodes[id].feat] < tree.nodes[id].cond
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
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    end
    return pred
end

# prediction in Leaf - GradientRegression
function pred_leaf(loss::S, node::TrainNode, params::EvoTreeRegressor, δ²) where {S<:GradientRegression, T<:AbstractFloat}
    pred = - params.η * node.∑δ / (node.∑δ² + params.λ * node.∑𝑤)
    return pred
end

# prediction in Leaf - L1Regression
function pred_leaf(loss::S, node::TrainNode, params::EvoTreeRegressor, δ²) where {S<:L1Regression, T<:AbstractFloat}
    pred = params.η * node.∑δ / (node.∑𝑤 * (1+params.λ))
    return pred
end

# prediction in Leaf - QuantileRegression
function pred_leaf(loss::S, node::TrainNode, params::EvoTreeRegressor, δ²) where {S<:QuantileRegression, T<:AbstractFloat}
    pred = params.η * quantile(δ²[collect(node.𝑖)], params.α) / (1 + params.λ)
    return pred
end
