# define an abstrat tree node type - concrete types are TreeSplit and TreeLeaf
abstract type TreeNode{T<:AbstractFloat} end

# object containing data-wise info
# X, Y, and gradients are pre-sorted to speedup training
# gradients are to be updated through training
struct ModelData{T<:AbstractFloat}
    X::Matrix{T}
    X_permsort::Matrix{T}
    Y::Vector{T}
    δ::Vector{T}
    δ²::Vector{T}
    λ::T
end

# compact alternative to ModeLData - not used for now
# To Do: how to exploit pre-sorting and binning
struct TrainData{T<:AbstractFloat}
    X::Matrix{T}
    δ::Vector{T}
    δ²::Vector{T}
end


mutable struct SplitInfo{T<:AbstractFloat}
    gain::T
    ∑δL::T
    ∑δ²L::T
    ∑δR::T
    ∑δ²R::T
    gainL::T
    gainR::T
    𝑖L::Vector
    𝑖R::Vector
    feat::Int
    cond::T
end

mutable struct SplitInfo2{T<:AbstractFloat}
    gain::T
    ∑δL::T
    ∑δ²L::T
    ∑δR::T
    ∑δ²R::T
    gainL::T
    gainR::T
    𝑖::Int
    feat::Int
    cond::T
end

mutable struct SplitTrack{T<:AbstractFloat}
    ∑δL::T
    ∑δ²L::T
    ∑δR::T
    ∑δ²R::T
    gainL::T
    gainR::T
    gain::T
end

mutable struct TreeLeaf{T<:AbstractFloat} <: TreeNode{T}
    depth::Int
    ∑δ::T
    ∑δ²::T
    gain::T
    pred::T
end

mutable struct TreeSplit{T<:AbstractFloat} <: TreeNode{T}
    left::TreeNode
    right::TreeNode
    feat::Int
    cond::T
end

struct Params{T<:AbstractFloat}
    loss::Symbol
    nrounds::Int
    λ::T
    γ::T
    η::T
    max_depth::Int
    min_weight::T
    rowsample::T
    colsample::T
end

# gradient-boosted tree is formed by a vector of trees
struct GBTree
    trees::Vector{TreeNode}
    params::Params
end


############################
# Vectorized approach
############################

# single tree is made of a root node that containes nested nodes and leafs
mutable struct Node
    depth
    ∑δ
    ∑δ²
    gain
    feat
    cond
    left
    right
    pred
    𝑖::Vector{Int}
end

# single tree is made of a root node that containes nested nodes and leafs
struct Tree
    nodes::Vector{Node}
end

struct GBTrees
    trees::Vector{Tree}
    params::Params
end
