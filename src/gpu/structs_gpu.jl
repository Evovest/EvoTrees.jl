# store perf info of each variable
mutable struct SplitInfo_gpu{T<:AbstractFloat, S<:Int}
    gain::T
    ∑δL::Vector{T}
    ∑δ²L::Vector{T}
    ∑𝑤L::T
    ∑δR::Vector{T}
    ∑δ²R::Vector{T}
    ∑𝑤R::T
    gainL::T
    gainR::T
    𝑖::S
    feat::S
    cond::T
end

struct TreeNode_gpu{T<:AbstractFloat, S<:Int, B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    gain::T
    pred::Vector{T}
    split::B
end

TreeNode_gpu(left::S, right::S, feat::S, cond::T, gain::T, K) where {T<:AbstractFloat, S<:Int} = TreeNode_gpu(left, right, feat, cond, gain, zeros(T,K), true)
TreeNode_gpu(pred::Vector{T}) where {T} = TreeNode_gpu(0, 0, 0, zero(T), zero(T), pred, false)

# single tree is made of a root node that containes nested nodes and leafs
struct TrainNode_gpu{T<:AbstractFloat, S<:Int, V<:AbstractVector}
    parent::S
    depth::S
    ∑δ::Vector{T}
    ∑δ²::Vector{T}
    ∑𝑤::T
    gain::T
    𝑖::V
    𝑗::V
end

# single tree is made of a root node that containes nested nodes and leafs
struct Tree_gpu{T<:AbstractFloat, S<:Int}
    nodes::Vector{TreeNode_gpu{T,S,Bool}}
end

# gradient-boosted tree is formed by a vector of trees
struct GBTree_gpu{T<:AbstractFloat, S<:Int}
    trees::Vector{Tree_gpu{T,S}}
    params::EvoTypes
    metric::Metric
    K::Int
    levels
end
