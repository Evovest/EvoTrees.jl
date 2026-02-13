module Shap

export shap

using ..EvoTrees
using .Threads: @threads

struct ShapTree
    weights::Vector{Float64}
    leaf_predictions::Vector{Float64}
    parents::Vector{Int}
    edge_heights::Vector{Int}
    features::Vector{Int}
    children_left::Vector{Int}
    children_right::Vector{Int}
    thresholds::Vector{UInt8}
    max_depth::Int
    num_nodes::Int
end

"""
    ShapTree(tree::EvoTrees.Tree)

Copy an EvoTrees.Tree into a struct compatible with the linear-tree-shap algorithm.
"""
function ShapTree(tree::EvoTrees.Tree)
    n = length(tree.feat)
    weights = ones(Float64, n)
    leaf_predictions = zeros(Float64, n)
    parents = fill(-1, n)
    edge_heights = zeros(Int, n)
    # keep raw feature indices (0 indicates leaf/no feature)
    features = collect(tree.feat)
    children_left = similar(tree.feat)
    children_right = similar(tree.feat)
    thresholds = zeros(UInt8, n)

    # EvoTrees uses 1-based indexing for nodes and stores splits in arrays sized 2^depth-1
    # We'll adapt traversal using the convention in predict.jl: root node index = 1
    # children indices: left = nid << 1, right = (nid << 1) + 1
    for i = 1:n
        if tree.split[i]
            children_left[i] = (i << 1)
            children_right[i] = (i << 1) + 1
            thresholds[i] = tree.cond_bin[i]
        else
            children_left[i] = -1
            children_right[i] = -1
        end
    end

    # Use node sample counts from EvoTrees `w` field when available
    n_node_samples = Float64.(tree.w)

    # compute weights and parents and edge_heights following the python reference semantics
    function _recursive_copy(node::Int; feature::Int=0, parent_samples::Float64=n_node_samples[1], prod_weight=1.0, seen_features=Dict{Int,Int}())
        if node < 1 || node > n
            return 0
        end
        n_sample = n_node_samples[node]
        if feature != 0
            weight = n_sample / parent_samples
            prod_weight *= weight
            if haskey(seen_features, feature)
                parents[node] = seen_features[feature]
                weight *= weights[seen_features[feature]]
            end
            weights[node] = weight
            seen_features[feature] = node
        end

        left = children_left[node]
        right = children_right[node]
        if left >= 0
            left_max = _recursive_copy(left; feature=tree.feat[node], parent_samples=n_sample, prod_weight=prod_weight, seen_features=copy(seen_features))
            right_max = _recursive_copy(right; feature=tree.feat[node], parent_samples=n_sample, prod_weight=prod_weight, seen_features=copy(seen_features))
            edge_heights[node] = max(left_max, right_max)
            return edge_heights[node]
        else
            # leaf: height = number of distinct seen features
            edge_heights[node] = length(seen_features)
            root_samples = n_node_samples[1]
            leaf_predictions[node] = tree.pred[1, node] * (n_sample / root_samples)
            return edge_heights[node]
        end
    end

    _recursive_copy(1)

    # Determine tree max depth from the maximum split node index: depth = floor(log2(max_split_idx)) + 1
    split_indices = findall(tree.split)
    if isempty(split_indices)
        depth_from_splits = 0
    else
        max_split_idx = maximum(split_indices)
        depth_from_splits = Int(floor(log2(max_split_idx))) + 1
    end
    return ShapTree(weights, leaf_predictions, parents, edge_heights, features, children_left, children_right, thresholds, depth_from_splits, n)
end

"""
    inference(tree::ShapTree, X::AbstractMatrix{<:Real})

A port of the linear_tree_shap `inference_v2`.

# Arguments
- tree: ShapTree from `copy_tree`
- X: Matrix{UInt8} observations (rows are samples, columns features)

Returns: Matrix of shap values (n_samples x n_features)
"""
function inference(tree::ShapTree, X::AbstractMatrix{UInt8})
    out = zeros(Float64, size(X))
    inference!(out::AbstractMatrix, tree::ShapTree, X::AbstractMatrix{UInt8})
    return out
end

function inference!(out::AbstractMatrix, tree::ShapTree, X::AbstractMatrix{UInt8})
    # allocate one extra slot so cached matrices have room for indexes up to max_depth
    m = max(1, tree.max_depth + 1)
    # build D points (Chebyshev nodes) and precompute Vandermonde-like matrices
    D = chebpts2(m)
    # compute D_powers cache as in Python: np.vander(D+1).T[::-1]
    V = vander(D .+ 1)'
    D_powers = V[end:-1:1, :]
    Ns = get_N(D)

    # per-sample iteration
    @threads for i = axes(X, 1)
        activation = falses(tree.num_nodes)
        activation[1] = true  # root node is always active
        x = view(X, i, :)
        out_row = view(out, i, :)
        C = zeros(Float64, m, m)
        E = zeros(Float64, m, m)
        C[1, :] .= 1.0
        _inference!(out_row, tree, x, activation, D, D_powers, Ns, C, E, 1, -1, 0)
    end
    return nothing
end


"""
    shap(m::EvoTree, data; ntree_limit=length(m.trees))

Returns the shap effect as a Matrix of size `[nobs, features]`.

It's based on an implementation of Linear TreeShap by Yu et al. (2022). It computes exact Shapley values for decision trees in O(LD) time.
It was originally ported from this [repo](https://github.com/yupbank/linear_tree_shap).

# References

Peng Yu, Chao Xu, Albert Bifet, Jesse Read Linear Tree Shap (2022). In: [Proceedings of 36th Conference on Neural Information Processing Systems](https://openreview.net/forum?id=OzbkiUo24g).
"""
function shap(m::EvoTree, data; ntree_limit=length(m.trees))
    x_bin = EvoTrees.binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    out = zeros(Float64, size(x_bin))
    for i in 1:ntree_limit
        tree = ShapTree(m.trees[i])
        inference!(out, tree, x_bin)
    end
    return out
end


function chebpts2(n::Int)
    # map to python's np.polynomial.chebyshev.chebpts2
    if n <= 0
        return Float64[]
    end
    k = 0:(n-1)
    return cos.((2k .+ 1) .* pi ./ (2n))
end

function vander(x::AbstractVector{T}) where T
    n = length(x)
    V = zeros(Float64, n, n)
    for j in 1:n
        p = n - j
        V[:, j] .= Float64.(x) .^ p
    end
    return V
end

function get_N(D::AbstractVector{<:Real})
    depth = length(D)
    Ns = zeros(Float64, depth + 1, depth)
    for i = 1:depth
        if i == 0
            continue
        end
        V = vander(D[1:i])'
        Ns[i+0, 1:i] = V \ (1.0 ./ get_norm_weight(i - 1))
    end
    return Ns
end

function get_norm_weight(M::Int)
    w = zeros(Float64, M + 1)
    for i = 0:M
        w[i+1] = binomial(M, i)
    end
    return w
end

function psi(E_row::AbstractVector{T}, D_power::AbstractVector{<:Real}, D::AbstractVector{<:Real}, q, Ns, d) where T
    if d <= 0
        return 0.0
    end
    @views nvec = Ns[d, 1:d]
    @views vals = (E_row[1:d] .* D_power[1:d]) ./ (D[1:d] .+ q)
    return (vals' * nvec) / d
end

function _inference!(out_row, tree::ShapTree, x, activation, D::AbstractVector{<:Real}, D_powers::AbstractMatrix{<:Real}, Ns, C, E, node::Int, edge_feature::Int, depth::Int)
    left = tree.children_left[node]
    right = tree.children_right[node]
    parent = tree.parents[node]
    child_edge_feature = tree.features[node]

    left_height = left >= 1 ? tree.edge_heights[left] : 0
    right_height = right >= 1 ? tree.edge_heights[right] : 0
    parent_height = parent >= 1 ? tree.edge_heights[parent] : 0
    current_height = tree.edge_heights[node]

    # map python depth (0-based) to Julia row index
    idx = depth + 1

    # Set activation based on parent's activation first (matches C++ implementation)
    if parent >= 1
        activation[node] = activation[node] & activation[parent]
    end

    if left >= 1
        if child_edge_feature >= 1 && child_edge_feature <= length(x)
            if x[child_edge_feature] <= tree.thresholds[node]
                activation[left] = true
                activation[right] = false
            else
                activation[left] = false
                activation[right] = true
            end
        else
            # if feature index invalid, default left active
            activation[left] = true
            activation[right] = false
        end
    end

    if edge_feature >= 0
        q_eff = activation[node] ? 1.0 / tree.weights[node] : 0.0
        @views C[idx, :] .= C[idx-1, :] .* (D .+ q_eff)
        if parent >= 1
            s_eff = activation[parent] ? 1.0 / tree.weights[parent] : 0.0
            @views C[idx, :] .= C[idx, :] ./ (D .+ s_eff)
        end
    end

    if left < 1
        @views E[idx, :] .= C[idx, :] .* tree.leaf_predictions[node]
    else
        _inference!(out_row, tree, x, activation, D, D_powers, Ns, C, E, left, child_edge_feature, depth + 1)
        # E[depth] = E[depth+1]*Offset[current_height-left_height]
        delta1 = current_height - left_height + 1  # +1 for 1-based indexing
        row1 = clamp(delta1, 1, size(D_powers, 1))
        @views E[idx, :] .= E[idx+1, :] .* D_powers[row1, :]

        _inference!(out_row, tree, x, activation, D, D_powers, Ns, C, E, right, child_edge_feature, depth + 1)
        # E[depth] += E[depth+1]*Offset[current_height-right_height] 
        delta2 = current_height - right_height + 1  # +1 for 1-based indexing
        row2 = clamp(delta2, 1, size(D_powers, 1))
        @views E[idx, :] .+= E[idx+1, :] .* D_powers[row2, :]
    end

    if edge_feature >= 0
        # Early return if parent is inactive
        if parent >= 1 && !activation[parent]
            return
        end

        @views val = (q_eff - 1.0) * psi(E[idx, :], D_powers[1, :], D, q_eff, Ns, current_height)
        if edge_feature >= 1 && edge_feature <= length(out_row)
            out_row[edge_feature] += val
        end
        if parent >= 1
            s_eff = activation[parent] ? 1.0 / tree.weights[parent] : 0.0
            # Python: value = (s_eff-1)*psi_v2(E[depth], Offset[parent_height-current_height], Base, s_eff, Ns, parent_height)
            offset_row = parent_height - current_height + 1  # +1 for 1-based indexing
            offset_row = clamp(offset_row, 1, size(D_powers, 1))
            @views val2 = (s_eff - 1.0) * psi(E[idx, :], D_powers[offset_row, :], D, s_eff, Ns, parent_height)
            if edge_feature >= 1 && edge_feature <= length(out_row)
                out_row[edge_feature] -= val2
            end
        end
    end
end

end # module
