"""
    Carries training information for a given tree node
"""
mutable struct TrainNodeGPU{T<:AbstractFloat}
    gain::T
    is::Union{Nothing,AbstractVector{UInt32}}
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
        CUDA.zeros(T, nbins, nvars),
    )
    return node
end

"""
    single tree is made of a a vector of nodes
"""
struct TreeGPU{L,K,T<:AbstractFloat}
    feat::CuVector{Int}
    cond_bin::CuVector{UInt8}
    cond_float::CuVector{T}
    gain::CuVector{T}
    pred::CuMatrix{T}
    split::CuVector{Bool}
end

TreeGPU{L,K,T}(x::CuVector{T}) where {L,K,T} = TreeGPU{L,K,T}(
    CUDA.zeros(Int, 1),
    CUDA.zeros(UInt8, 1),
    CUDA.zeros(T, 1),
    CUDA.zeros(T, 1),
    reshape(x, :, 1),
    CUDA.zeros(Bool, 1),
)
TreeGPU{L,K,T}(depth::Int) where {L,K,T} = TreeGPU{L,K,T}(
    CUDA.zeros(Int, 2^depth - 1),
    CUDA.zeros(UInt8, 2^depth - 1),
    CUDA.zeros(T, 2^depth - 1),
    CUDA.zeros(T, 2^depth - 1),
    CUDA.zeros(T, K, 2^depth - 1),
    CUDA.zeros(Bool, 2^depth - 1),
)

# gradient-boosted tree is formed by a vector of trees
struct EvoTreeGPU{L,K,T}
    trees::Vector{TreeGPU{L,K,T}}
    info::Any
end
(m::EvoTreeGPU)(x::AbstractMatrix) = predict(m, x)
get_types(::EvoTreeGPU{L,K,T}) where {L,K,T} = (L,T)
