struct CacheBaseGPU{Y,N} <: EvoTrees.CacheGPU
    K::UInt8
    x_bin::CuMatrix{UInt8}
    y::Y
    w::Vector{Float32}
    pred::CuMatrix{Float32}
    nodes::N
    is_in::CuVector{UInt32}
    is_out::CuVector{UInt32}
    mask::CuVector{UInt8}
    js_::Vector{UInt32}
    js::Vector{UInt32}
    out::CuVector{UInt32}
    left::CuVector{UInt32}
    right::CuVector{UInt32}
    ∇::CuArray{Float32}
    h∇::CuArray{Float64, 3}
    h∇_cpu::Array{Float64, 3}
    feature_names::Vector{Symbol}
    featbins::Vector{UInt8}
    feattypes::Vector{Bool}
    feattypes_gpu::CuVector{Bool}
    monotone_constraints::Vector{Int32}
end
