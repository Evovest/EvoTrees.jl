"""
    store perf info of each variable
"""
struct SplitInfoGPU{V,M,B}
    gains::V
    gainsL::V
    gainsR::V
    ∑Ls::M
    ∑Rs::M
    bins::B
end

"""
    Carries training information for a given tree node
"""
struct TrainNodeGPU{T<:AbstractFloat, S, I<:AbstractVector, V<:AbstractVector}
    parent::S
    depth::S
    ∑::V
    gain::T
    𝑖::I
    𝑗::I
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
