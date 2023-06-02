"""
    Carries training information for a given tree node
"""
# mutable struct TrainNodeGPU{T<:AbstractFloat,S}
#     gain::T
#     is::S
#     ∑::AbstractVector{T}
#     h::AbstractArray{T,3}
#     hL::AbstractArray{T,3}
#     hR::AbstractArray{T,3}
#     gains::AbstractMatrix{T}
# end

function TrainNodeGPU(featbins, K, is, T)
    node = TrainNode(
        zero(T),
        is,
        CUDA.zeros(T, 2 * K + 1),
        [CUDA.zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [CUDA.zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [CUDA.zeros(T, 2 * K + 1, nbins) for nbins in featbins],
        [CUDA.zeros(T, nbins) for nbins in featbins],
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

function Base.show(io::IO, tree::TreeGPU)
    println(io, "$(typeof(tree))")
    for fname in fieldnames(typeof(tree))
        println(io, " - $fname: $(getfield(tree, fname))")
    end
end

# gradient-boosted tree is formed by a vector of trees
struct EvoTreeGPU{L,K,T}
    trees::Vector{TreeGPU{L,K,T}}
    info::Any
end
(m::EvoTreeGPU)(x::AbstractMatrix) = predict(m, x)
get_types(::EvoTreeGPU{L,K,T}) where {L,K,T} = (L, T)

function Base.show(io::IO, evotree::EvoTreeGPU)
    println(io, "$(typeof(evotree))")
    println(io, " - Contains $(length(evotree.trees)) trees in field `trees` (incl. 1 bias tree).")
    println(io, " - Data input has $(length(evotree.info[:fnames])) features.")
    println(io, " - $(keys(evotree.info)) info accessible in field `info`")
end