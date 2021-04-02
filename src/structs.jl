# define an abstrat tree node type - concrete types are TreeSplit and TreeLeaf
abstract type Node{T<:AbstractFloat} end


struct TreeNode{T<:AbstractFloat, S<:Integer, B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    gain::T
    pred::Vector{T}
    split::B
end

TreeNode(left::S, right::S, feat::S, cond::T, gain::T, L::S) where {T<:AbstractFloat, S<:Integer} = TreeNode{L,T,S,Bool}(left, right, feat, cond, gain, zeros(T, L), true)
TreeNode(pred::Vector{T}) where {T} = TreeNode(0, 0, 0, zero(T), zero(T), pred, false)

# single tree is made of a root node that containes nested nodes and leafs
struct TrainNode{T<:AbstractFloat, S<:Integer, V<:AbstractVector}
    parent::S
    depth::S
    ∑::V
    gain::T
end

# single tree is made of a root node that containes nested nodes and leafs
struct Tree{T<:AbstractFloat, S<:Int}
    nodes::Vector{TreeNode{T,S,Bool}}
end

# eval metric tracking
mutable struct Metric
    iter::Int
    metric::Float32
end
Metric() = Metric(0, Inf)

# gradient-boosted tree is formed by a vector of trees
struct GBTree{T<:AbstractFloat, S<:Int}
    trees::Vector{Tree{T,S}}
    params::EvoTypes
    metric::Metric
    K::Int
    levels
end
