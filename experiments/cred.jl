### compare mse vs cred based metrics
using DataFrames
using Statistics: mean, std
using CairoMakie

function gain_grad(∑::AbstractVector, lambda)
    ∑[1]^2 / (∑[2] + lambda * ∑[3]) / 2
end

function gain_cred(∑::AbstractVector, lambda)
    # ∑[1]^2 / (∑[2] + lambda * ∑[3]) / 2
    ∑[1] .* ∑[3]
end

# MSE
function update_grads(p::Vector{T}, y::Vector{T}) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    for i in eachindex(y)
        ∇[1, i] = 2 * (p[i] - y[i]) * ∇[3, i]
        ∇[2, i] = 2 * ∇[3, i]
    end
    return ∇
end

# Cred
function update_creds(p::Vector{T}, y::Vector{T}) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    for i in eachindex(y)
        ∇[1, i] = (p[i] - y[i]) * ∇[3, i]
        ∇[2, i] = (p[i] - y[i]) .^ 2 * ∇[3, i]
    end
    return ∇
end

################################################
# data setup
################################################
nobs = 1000
σ = 2.0
yL, yR = rand(nobs), rand(nobs)
yL .= (yL .- mean(yL)) ./ std(yL) .* σ .- 0.5
yR .= (yR .- mean(yR)) ./ std(yR) .* σ .+ 0.5
yT = vcat(yL, yR)
mean(yT)
std(yT)

pL = zero(yL)
pR = zero(yR)
pT = zero(yT)

# f = Figure()
f, ax = density(yL)
density!(ax, yR)
f

################################################
# MSE grads: 
#   - linear gain with nobs
#   - invariant to sigma
################################################
∇gT = update_grads(pT, yT)
∑gT = dropdims(sum(∇gT, dims=2), dims=2)
@info gT = gain_grad(∑gT, 0.0)

∇gL = update_grads(pL, yL)
∑gL = dropdims(sum(∇gL, dims=2), dims=2)
@info gL = gain_grad(∑gL, 0)

∇gR = update_grads(pR, yR)
∑gR = dropdims(sum(∇gR, dims=2), dims=2)
@info gR = gain_grad(∑gR, 0)

################################################
# Cred: 
#   - linear gain with nobs
#   - invariant to sigma
################################################
∇cT = update_creds(pT, yT)
∑cT = dropdims(sum(∇cT, dims=2), dims=2)
@info cT = gain_grad(∑cT, 0.0)

∇cL = update_creds(pL, yL)
∑cL = dropdims(sum(∇cL, dims=2), dims=2)
@info cL = gain_grad(∑cL, 0)

∇cR = update_creds(pR, yR)
∑cR = dropdims(sum(∇cR, dims=2), dims=2)
@info cR = gain_grad(∑cR, 0)
