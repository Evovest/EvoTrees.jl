"""
    store perf info of each variable - looking to drop
"""
mutable struct SplitInfoGPU{T<:AbstractFloat, S}
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

"""
    Carries training information for a given tree node
"""
struct TrainNodeGPU{T<:AbstractFloat, S, V<:AbstractVector}
    parent::S
    depth::S
    ∑δ::Vector{T}
    gain::T
    𝑖::V
    𝑗::V
end

struct TreeNodeGPU{T<:AbstractFloat, S, B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    gain::T
    pred::Vector{T}
    split::B
end

TreeNodeGPU(left::S, right::S, feat::S, cond::T, gain::T, K) where {T<:AbstractFloat, S} = TreeNodeGPU(left, right, feat, cond, gain, zeros(T,K), true)
TreeNodeGPU(pred::Vector{T}) where {T} = TreeNodeGPU(UInt32(0), UInt32(0), UInt32(0), zero(T), zero(T), pred, false)

"""
    single tree is made of a a vector of nodes
"""
struct TreeGPU{T<:AbstractFloat, S}
    nodes::Vector{TreeNodeGPU{T,S,Bool}}
end

# gradient-boosted tree is formed by a vector of trees
struct GBTreeGPU{T<:AbstractFloat, S}
    trees::Vector{TreeGPU{T,S}}
    params::EvoTypes
    metric::Metric
    K::S
    levels
end
