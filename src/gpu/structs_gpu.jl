"""
    Carries training information for a given tree node
"""
mutable struct TrainNodeGPU{T<:AbstractFloat}
    gain::T
    𝑖::Union{Nothing, AbstractVector{UInt32}}
    ∑::Vector{T}
    h::Matrix{T}
    hL::Matrix{T}
    hR::Matrix{T}
    gains::Matrix{T}
end

function TrainNodeGPU(nvars, nbins, K, T)
    node = TrainNode{T}(
            zero(T),
            nothing,
            CUDA.zeros(T, 2*K+1), 
            CUDA.zeros(T, (2*K+1) * nbins, nvars), 
            CUDA.zeros(T, (2*K+1) * nbins, nvars), 
            CUDA.zeros(T, (2*K+1) * nbins, nvars), 
            CUDA.zeros(T, nbins, nvars))
    return node
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
struct TreeGPU{T<:AbstractFloat}
    feat::CuVector{Int}
    cond_bin::CuVector{UInt8}
    cond_float::CuVector{T}
    gain::CuVector{T}
    pred::CuMatrix{T}
    split::CuVector{Bool}
end

TreeGPU(x::CuVector{T}) where T <: AbstractFloat = TreeGPU(CUDA.zeros(Int, 1), CUDA.zeros(UInt8, 1), CUDA.zeros(T, 1), CUDA.zeros(T, 1), reshape(x, :, 1), CUDA.zeros(Bool, 1))
TreeGPU(depth::S, K::S, ::T) where {S <: Integer, T <: AbstractFloat} = TreeGPU(CUDA.zeros(Int, 2^depth-1), CUDA.zeros(UInt8, 2^depth-1), CUDA.zeros(T, 2^depth-1), CUDA.zeros(T, 2^depth-1), CUDA.zeros(T, K, 2^depth-1), CUDA.zeros(Bool, 2^depth-1))

# gradient-boosted tree is formed by a vector of trees
struct GBTreeGPU{T<:AbstractFloat, S<:Integer}
    trees::Vector{TreeGPU{T}}
    params::EvoTypes
    metric::Metric
    K::S
    levels
end
