"""
    Carries training information for a given tree node
"""
mutable struct TrainNode{T<:AbstractFloat,S,V,M}
    gain::T
    is::S
    ∑::V
    h::Vector{M}
    hL::Vector{M}
    hR::Vector{M}
    gains::Vector{V}
end

function TrainNode(featbins, K, is, T)
    node = TrainNode(
        zero(T),
        is,
        zeros(T, 2 * K + 1),
        [zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [zeros(T, nbins) for nbins in featbins],
    )
    return node
end

# single tree is made of a vectors of length num nodes
struct Tree{L,K,T}
    feat::Vector{Int}
    cond_bin::Vector{UInt8}
    cond_float::Vector{Any}
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

function Base.show(io::IO, tree::Tree)
    println(io, "$(typeof(tree))")
    for fname in fieldnames(typeof(tree))
        println(io, " - $fname: $(getfield(tree, fname))")
    end
end

"""
    EvoTree{L,K,T}

An EvoTree holds the structure of a fitted gradient-boosted tree.

# Fields
- trees::Vector{Tree{L,K,T}}
- info::Dict

`EvoTree` acts as a functor to perform inference on input data: 
```
pred = (m::EvoTree; ntree_limit=length(m.trees))(x)
```
"""
struct EvoTree{L,K,T}
    trees::Vector{Tree{L,K,T}}
    info::Dict
end
(m::EvoTree)(x::AbstractMatrix; ntree_limit=length(m.trees)) = predict(m, x; ntree_limit)
(m::EvoTree)(df::AbstractDataFrame; ntree_limit=length(m.trees)) = predict(m, df; ntree_limit)
get_types(::EvoTree{L,K,T}) where {L,K,T} = (L, T)

function Base.show(io::IO, evotree::EvoTree)
    println(io, "$(typeof(evotree))")
    println(io, " - Contains $(length(evotree.trees)) trees in field `trees` (incl. 1 bias tree).")
    println(io, " - Data input has $(length(evotree.info[:fnames])) features.")
    println(io, " - $(keys(evotree.info)) info accessible in field `info`")
end