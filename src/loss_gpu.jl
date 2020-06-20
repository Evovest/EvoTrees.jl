# linear
function update_grads_gpu!(loss::Linear, pred::AbstractVector{T}, target::AbstractVector{T}, δ::AbstractVector{T}, δ²::AbstractVector{T}, 𝑤::AbstractVector{T}) where {T <: AbstractFloat}
    @. δ = 2f0 * (pred - target) * 𝑤
    @. δ² = 2f0 * 𝑤
    return
end

# linear
function get_gain(loss::S, ∑δ::T, ∑δ²::T, ∑𝑤::T, λ::T) where {S <: GradientRegression, T <: AbstractFloat}
    gain = (∑δ ^ 2 / (∑δ² + λ * ∑𝑤)) / 2
    return gain
end
