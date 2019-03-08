# define an abstrat tree node type - concrete types are TreeSplit and TreeLeaf
abstract type Node{T<:AbstractFloat} end

# compact alternative to ModeLData - not used for now
# To Do: how to exploit pre-sorting and binning
struct TrainData{T<:AbstractFloat}
    X::Matrix{T}
    X_permsort::Matrix{T}
    Y::Matrix{T}
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

struct TreeNode{T<:AbstractFloat, S<:Int}
    left::S
    right::S
    feat::S
    cond::T
    pred::T
    split::Bool
end

TreeNode(left::S, right::S, feat::S, cond::T) where {S<:Int, T<:AbstractFloat} = TreeNode(left, right, feat, cond, 0.0, true)
TreeNode(pred::T) where {T<:AbstractFloat} = TreeNode(0, 0, 0, 0.0, pred, false)

struct Params{T<:AbstractFloat}
    loss::Symbol
    nrounds::Int
    λ::T
    γ::T
    η::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T # colsample_bytree
end

# single tree is made of a root node that containes nested nodes and leafs
struct TrainNode{T<:AbstractFloat, I<:AbstractArray{Int, 1}, J<:AbstractArray{Int, 1}, S<:Int}
    depth::S
    ∑δ::T
    ∑δ²::T
    gain::T
    𝑖::I
    𝑗::J
end

# single tree is made of a root node that containes nested nodes and leafs
struct Tree{T, S}
    nodes::Vector{TreeNode{T, S}}
end

# gradient-boosted tree is formed by a vector of trees
struct GBTree
    trees::Vector{Tree}
    params::Params
end
