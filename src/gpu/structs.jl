# gradient-boosted tree is formed by a vector of trees
struct EvoTreeGPU{L,K,T}
    trees::Vector{Tree{L,K,T}}
    info::Dict
end
(m::EvoTreeGPU)(data; ntree_limit=length(m.trees)) = predict(m, data; ntree_limit)
get_types(::EvoTreeGPU{L,K,T}) where {L,K,T} = (L, T)

function Base.show(io::IO, evotree::EvoTreeGPU)
    println(io, "$(typeof(evotree))")
    println(io, " - Contains $(length(evotree.trees)) trees in field `trees` (incl. 1 bias tree).")
    println(io, " - Data input has $(length(evotree.info[:fnames])) features.")
    println(io, " - $(keys(evotree.info)) info accessible in field `info`")
end