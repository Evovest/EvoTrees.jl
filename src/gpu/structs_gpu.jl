"""
    Carries training information for a given tree node
"""
mutable struct TrainNodeGPU{T<:AbstractFloat}
    gain::T
    𝑖::Union{Nothing,AbstractVector{UInt32}}
    ∑::AbstractVector{T}
    h::AbstractArray{T,3}
    hL::AbstractArray{T,3}
    hR::AbstractArray{T,3}
    gains::AbstractMatrix{T}
end

function TrainNodeGPU(nvars, nbins, K, T)
    node = TrainNodeGPU{T}(
        zero(T),
        nothing,
        CUDA.zeros(T, 2 * K + 1),
        CUDA.zeros(T, (2 * K + 1), nbins, nvars),
        CUDA.zeros(T, (2 * K + 1), nbins, nvars),
        CUDA.zeros(T, (2 * K + 1), nbins, nvars),
        CUDA.zeros(T, nbins, nvars))
    return node
end


struct TreeNodeGPU{T<:AbstractFloat,S,B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    gain::T
    pred::Vector{T}
    split::B
end

TreeNodeGPU(left::S, right::S, feat::S, cond::T, gain::T, K) where {T<:AbstractFloat,S} = TreeNodeGPU(left, right, feat, cond, gain, zeros(T, K), true)
TreeNodeGPU(pred::Vector{T}) where {T} = TreeNodeGPU(UInt32(0), UInt32(0), UInt32(0), zero(T), zero(T), pred, false)

"""
    single tree is made of a a vector of nodes
"""
struct TreeGPU{L,T<:AbstractFloat}
    feat::CuVector{Int}
    cond_bin::CuVector{UInt8}
    cond_float::CuVector{T}
    gain::CuVector{T}
    pred::CuMatrix{T}
    split::CuVector{Bool}
end

TreeGPU{L,T}(x::CuVector{T}) where {L,T} = TreeGPU{L,T}(CUDA.zeros(Int, 1), CUDA.zeros(UInt8, 1), CUDA.zeros(T, 1), CUDA.zeros(T, 1), reshape(x, :, 1), CUDA.zeros(Bool, 1))
TreeGPU{L,T}(depth, K, ::T) where {L,T} = TreeGPU{L,T}(CUDA.zeros(Int, 2^depth - 1), CUDA.zeros(UInt8, 2^depth - 1), CUDA.zeros(T, 2^depth - 1), CUDA.zeros(T, 2^depth - 1), CUDA.zeros(T, K, 2^depth - 1), CUDA.zeros(Bool, 2^depth - 1))

# gradient-boosted tree is formed by a vector of trees
struct GBTreeGPU{L,T,S}
    trees::Vector{TreeGPU{L,T}}
    params::EvoTypes{L,T,S}
    metric::Metric
    K::Int
    levels
end
(m::GBTreeGPU)(x::AbstractMatrix) = predict(m, x)