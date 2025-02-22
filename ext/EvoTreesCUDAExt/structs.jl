mutable struct CacheBaseGPU{Y,N} <: EvoTrees.CacheGPU
    nrounds::UInt32
    const K::UInt8
    const x_bin::CuMatrix{UInt8}
    const y::Y
    const w::Vector{Float32}
    const pred::CuMatrix{Float32}
    const nodes::N
    const is_in::CuVector{UInt32}
    const is_out::CuVector{UInt32}
    const mask::CuVector{UInt8}
    const js_::Vector{UInt32}
    const js::Vector{UInt32}
    const out::CuVector{UInt32}
    const left::CuVector{UInt32}
    const right::CuVector{UInt32}
    const ∇::CuArray{Float32}
    const h∇::CuArray{Float64, 3}
    const h∇_cpu::Array{Float64, 3}
    const feature_names::Vector{Symbol}
    const featbins::Vector{UInt8}
    const feattypes::Vector{Bool}
    const feattypes_gpu::CuVector{Bool}
    const monotone_constraints::Vector{Int32}
end
