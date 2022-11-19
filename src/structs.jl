"""
    Carries training information for a given tree node
"""
mutable struct TrainNode{T<:AbstractFloat,S}
    gain::T
    is::S
    ∑::Vector{T}
    h::Array{T,3}
    hL::Array{T,3}
    hR::Array{T,3}
    gains::Matrix{T}
end

function TrainNode(nvars, nbins, K, is, T)
    node = TrainNode(
        zero(T),
        is,
        zeros(T, 2 * K + 1),
        zeros(T, 2 * K + 1, nbins, nvars),
        zeros(T, 2 * K + 1, nbins, nvars),
        zeros(T, 2 * K + 1, nbins, nvars),
        zeros(T, nbins, nvars),
    )
    return node
end

# single tree is made of a vectors of length num nodes
struct Tree{L,K,T}
    feat::Vector{Int}
    cond_bin::Vector{UInt8}
    cond_float::Vector{T}
    gain::Vector{T}
    pred::Matrix{T}
    split::Vector{Bool}
end

function Tree{L,K,T}(x::Vector) where {L,K,T}
    Tree{L,K,T}(
        zeros(Int, 1),
        zeros(UInt8, 1),
        zeros(T, 1),
        zeros(T, 1),
        reshape(x, :, 1),
        zeros(Bool, 1),
    )
end

function Tree{L,K,T}(depth::Int) where {L,K,T}
    Tree{L,K,T}(
        zeros(Int, 2^depth - 1),
        zeros(UInt8, 2^depth - 1),
        zeros(T, 2^depth - 1),
        zeros(T, 2^depth - 1),
        zeros(T, K, 2^depth - 1),
        zeros(Bool, 2^depth - 1),
    )
end

# gradient-boosted tree is formed by a vector of trees
struct EvoTree{L,K,T}
    trees::Vector{Tree{L,K,T}}
    info::Dict
end
(m::EvoTree)(x::AbstractMatrix) = predict(m, x)
get_types(::EvoTree{L,K,T}) where {L,K,T} = (L,T)
