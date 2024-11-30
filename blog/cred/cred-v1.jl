### compare mse vs cred based metrics
using DataFrames
using Distributions
using Statistics: mean, std
using CairoMakie

# MSE
function sum_grads(p::Vector{T}, y::Vector{T}) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    for i in eachindex(y)
        ∇[1, i] = 2 * (p[i] - y[i]) * ∇[3, i]
        ∇[2, i] = 2 * ∇[3, i]
    end
    ∑ = dropdims(sum(∇, dims=2), dims=2)
    return ∑
end
function gain_grad(∑::AbstractVector, lambda)
    return ∑[1]^2 / (∑[2] + lambda * ∑[3]) / 2
end

# Cred
function sum_creds(p::Vector{T}, y::Vector{T}) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    for i in eachindex(y)
        ∇[1, i] = (p[i] - y[i]) * ∇[3, i]
        ∇[2, i] = (p[i] - y[i])^2 * ∇[3, i]
    end
    ∑ = dropdims(sum(∇, dims=2), dims=2)
    return ∑
end
get_cred(∑::AbstractVector) = ∑[1]^2 / max(eps(eltype(∑)), ∑[1]^2 + ∑[2]) # var
# get_cred(∑::AbstractVector) = ∑[1]^2 / max(eps(eltype(∑)), ∑[1]^2 + ∑[3] * ∑[2]) # var
function gain_cred(∑::AbstractVector, lambda)
    Z = get_cred(∑::AbstractVector)
    return Z * abs(∑[1])
end

################################################
# data setup
################################################
nobs = 100
σ = 1.0
Δμ = 1.0
yL, yR = randn(nobs), randn(nobs)
yL .= (yL .- mean(yL)) ./ std(yL) .* σ .- Δμ / 2
yR .= (yR .- mean(yR)) ./ std(yR) .* σ .+ Δμ / 2
yT = vcat(yL, yR)
mean(yT)
std(yT)

pL = zero(yL)
pR = zero(yR)
pT = zero(yT)

μL = rand(Normal(-Δμ / 2, σ / sqrt(nobs)), 100_000)
μR = rand(Normal(Δμ / 2, σ / sqrt(nobs)), 100_000)

ax
f, ax = density(yL; color="#02723599", label="left")
density!(ax, yR; color="#02357299", label="right")

density!(ax, μL; color="#027235FF")
density!(ax, μR; color="#023572FF")
f

################################################
# MSE grads: 
#   - linear gain with nobs
#   - invariant to sigma
################################################
∑gT = sum_grads(pT, yT)
@info gT = gain_grad(∑gT, 0.0)

∑gL = sum_grads(pL, yL)
@info gL = gain_grad(∑gL, 0)

∑gR = sum_grads(pR, yR)
@info gR = gain_grad(∑gR, 0)

################################################
# Cred: 
#   - linear gain with nobs
#   - invariant to sigma
################################################
∑cT = sum_creds(pT, yT)
@info cT = gain_cred(∑cT, 0.0)

∑cL = sum_creds(pL, yL)
@info cL = gain_cred(∑cL, 0)

∑cR = sum_creds(pR, yR)
@info cR = gain_cred(∑cR, 0)

function get_data(; nobs, spread=1.0, sd=1.0, lambda=0.0)
    yL, yR = randn(nobs), randn(nobs)
    yL .= (yL .- mean(yL)) ./ std(yL) .* sd .- spread / 2
    yR .= (yR .- mean(yR)) ./ std(yR) .* sd .+ spread / 2
    yT = vcat(yL, yR)

    pL = zero(yL)
    pR = zero(yR)
    pT = zero(yT)

    # Grads
    data = Dict()
    data[:yL] = yL
    data[:yR] = yR

    # Grads
    ∑gT = sum_grads(pT, yT)
    ∑gL = sum_grads(pL, yL)
    ∑gR = sum_grads(pR, yR)
    data[:gT] = gain_grad(∑gT, lambda)
    data[:gL] = gain_grad(∑gL, lambda)
    data[:gR] = gain_grad(∑gR, lambda)

    # Creds
    ∑cT = sum_creds(pT, yT)
    ∑cL = sum_creds(pL, yL)
    ∑cR = sum_creds(pR, yR)
    data[:cT] = gain_cred(∑cT, lambda)
    data[:cL] = gain_cred(∑cL, lambda)
    data[:cR] = gain_cred(∑cR, lambda)
    data[:ZR] = get_cred(∑cR)

    return data

end

function get_figure(data)

    gL = round(data[:gL]; sigdigits=4)
    gR = round(data[:gR]; sigdigits=4)

    cL = round(data[:cL]; sigdigits=4)
    cR = round(data[:cR]; sigdigits=4)
    ZR = round(data[:ZR]; sigdigits=4)

    f = Figure()
    ax1 = Axis(f[1, 1];
        subtitle=
        """
        Grad: gainL=$gL | gainR=$gR\n
        Cred: gainL=$cL | gainR=$cR | ZR=$ZR
        """
    )
    density!(ax1, data[:yL]; color="#02723599", label="left")
    density!(ax1, data[:yR]; color="#02357299", label="right")
    Legend(f[2, 1], ax1, orientation=:horizontal)
    return f
end

d1 = get_data(; nobs=10, spread=1.0, sd=1.0, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=100, spread=1.0, sd=1.0, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=10000, spread=1.0, sd=1.0, lambda=0.0)
f1 = get_figure(d1)

d1 = get_data(; nobs=10, spread=0.1, sd=1.0, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=100, spread=0.1, sd=1.0, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=10000, spread=0.1, sd=1.0, lambda=0.0)
f1 = get_figure(d1)

d1 = get_data(; nobs=10, spread=0.1, sd=0.1, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=100, spread=0.1, sd=0.1, lambda=0.0)
f1 = get_figure(d1)
d1 = get_data(; nobs=10000, spread=0.1, sd=0.1, lambda=0.0)
f1 = get_figure(d1)

f = Figure()
