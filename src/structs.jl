"""
    Carries training information for a given tree node
"""
mutable struct TrainNode{T<:AbstractFloat}
    gain::T
    𝑖::Union{Nothing,AbstractVector{UInt32}}
    ∑::Vector{T}
    h::Vector{Vector{T}}
    hL::Vector{Vector{T}}
    hR::Vector{Vector{T}}
    gains::Matrix{T}
end

function TrainNode(nvars, nbins, K, T)
    node = TrainNode{T}(
        zero(T),
        nothing,
        zeros(T, 2 * K + 1),
        [zeros(T, (2 * K + 1) * nbins) for j = 1:nvars],
        [zeros(T, (2 * K + 1) * nbins) for j = 1:nvars],
        [zeros(T, (2 * K + 1) * nbins) for j = 1:nvars],
        zeros(T, nbins, nvars),
    )
    return node
end

# single tree is made of a vectors of length num nodes
struct Tree{L,T<:AbstractFloat}
    feat::Vector{Int}
    cond_bin::Vector{UInt8}
    cond_float::Vector{T}
    gain::Vector{T}
    pred::Matrix{T}
    split::Vector{Bool}
end

function Tree{L,T}(x::Vector{T}) where {L,T}
    Tree{L,T}(
        zeros(Int, 1),
        zeros(UInt8, 1),
        zeros(T, 1),
        zeros(T, 1),
        reshape(x, :, 1),
        zeros(Bool, 1),
    )
end

function Tree{L,T}(depth, K, ::T) where {L,T}
    Tree{L,T}(
        zeros(Int, 2^depth - 1),
        zeros(UInt8, 2^depth - 1),
        zeros(T, 2^depth - 1),
        zeros(T, 2^depth - 1),
        zeros(T, K, 2^depth - 1),
        zeros(Bool, 2^depth - 1),
    )
end

# eval metric tracking
mutable struct Metric
    iter::Int
    metric::Float32
end
Metric() = Metric(0, Inf)

# gradient-boosted tree is formed by a vector of trees
struct GBTree{L,T,S}
    trees::Vector{Tree{L,T}}
    params::EvoTypes
    metric::Metric
    K::Int
    info::Any
end
(m::GBTree)(x::AbstractMatrix) = predict(m, x)
