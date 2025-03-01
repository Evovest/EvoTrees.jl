abstract type Device end
abstract type CPU <: Device end
abstract type GPU <: Device end

"""
    TrainNode{S,V,M}
    
Carries training information for a given tree node
"""
mutable struct TrainNode{S,V,M}
    gain::Float64
    is::S
    ∑::V
    h::Vector{M}
    hL::Vector{M}
    hR::Vector{M}
    gains::Vector{V}
end

function TrainNode(featbins, K, is)
    node = TrainNode(
        zero(Float64),
        is,
        zeros(2 * K + 1),
        [zeros(2 * K + 1, nbins) for nbins in featbins],
        [zeros(2 * K + 1, nbins) for nbins in featbins],
        [zeros(2 * K + 1, nbins) for nbins in featbins],
        [zeros(nbins) for nbins in featbins],
    )
    return node
end

abstract type Cache end
abstract type CacheCPU <: Cache end
abstract type CacheGPU <: Cache end

struct CacheBaseCPU{Y,N} <: CacheCPU
    K::UInt8
    x_bin::Matrix{UInt8}
    y::Y
    w::Vector{Float32}
    pred::Matrix{Float32}
    nodes::N
    # is_in::Vector{UInt32}
    is_in::UnitRange{Int64}
    is_out::Vector{UInt32}
    mask::Vector{UInt8}
    js_::Vector{UInt32}
    js::Vector{UInt32}
    # out::Vector{UInt32}
    out::Vector{Bool}
    left::Vector{UInt32}
    right::Vector{UInt32}
    ∇::Matrix{Float32}
    feature_names::Vector{Symbol}
    featbins::Vector{UInt8}
    feattypes::Vector{Bool}
    monotone_constraints::Vector{Int32}
end

# single tree is made of a vectors of length num nodes
struct Tree{L,K}
    feat::Vector{Int}
    cond_bin::Vector{UInt8}
    gain::Vector{Float64}
    pred::Matrix{Float32}
    split::Vector{Bool}
end

function Tree{L,K}(x::Vector) where {L,K}
    Tree{L,K}(
        zeros(Int, 1),
        zeros(UInt8, 1),
        zeros(Float64, 1),
        reshape(x, :, 1),
        zeros(Bool, 1),
    )
end

function Tree{L,K}(depth::Int) where {L,K}
    Tree{L,K}(
        zeros(Int, 2^depth - 1),
        zeros(UInt8, 2^depth - 1),
        zeros(Float64, 2^depth - 1),
        zeros(Float32, K, 2^depth - 1),
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
    EvoTree{L,K}

An `EvoTree` holds the structure of a fitted gradient-boosted tree.

# Fields
- trees::Vector{Tree{L,K}}
- info::Dict

`EvoTree` acts as a functor to perform inference on input data: 
```
pred = (m::EvoTree; ntree_limit=length(m.trees))(x)
```
"""
struct EvoTree{L,K}
    loss_type::Type{L}
    K::UInt8
    trees::Vector{Tree{L,K}}
    info::Dict{Symbol,Any}
end
function (m::EvoTree)(data; ntree_limit=length(m.trees), device=:cpu)
    @assert Symbol(device) ∈ [:cpu, :gpu]
    _device = Symbol(device) == :cpu ? CPU : GPU
    return _predict(m, data, _device; ntree_limit)
end

function Base.show(io::IO, evotree::EvoTree)
    println(io, "$(typeof(evotree))")
    println(io, " - Contains $(length(evotree.trees)) trees in field `trees` (incl. 1 bias tree).")
    println(io, " - Data input has $(length(evotree.info[:feature_names])) features.")
    println(io, " - $(keys(evotree.info)) info accessible in field `info`")
end
