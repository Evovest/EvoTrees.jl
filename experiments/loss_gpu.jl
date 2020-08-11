using CUDAnative
using CuArrays
using StaticArrays
using BenchmarkTools

N = Int(1e6)
pred = rand(SVector{1, Float32}, N)
target = rand(Float32, N)
δ = zeros(SVector{1, Float32}, N)
δ² = zeros(SVector{1, Float32}, N)

pred_g = CuArray(rand(Float32, N))
target_g = CuArray(rand(Float32, N))
δ_g = CuArray(zeros(Float32, N))
δ²_g = CuArray(zeros(Float32, N))

pred = Array(pred_g)
target = Array(target_g)
δ = Array(δ_g)
δ² = Array(δ²_g)

# linear
function update_grads!(pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}) where {T <: AbstractFloat, L}
    @inbounds for i in eachindex(δ)
        δ[i] = SVector(2 * (pred[i][1] - target[i]))
        δ²[i] = SVector(2)
    end
end

# linear
function update_grads_gpu!(pred::AbstractVector{T}, target::AbstractVector{T}, δ::AbstractVector{T}, δ²::AbstractVector{T}, 𝑤::AbstractVector{T}) where {T <: AbstractFloat}
    @. δ = 2f0 * (pred - target) * 𝑤
    @. δ² = 2f0 * 𝑤
    return
end

@time update_grads!(pred, target, δ, δ²)
CuArrays.@time update_grads_gpu!(pred_g, target_g, δ_g, δ²_g)
