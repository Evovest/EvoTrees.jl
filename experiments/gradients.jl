using Revise
using EvoTrees
using Base.Threads

L = EvoTrees.Logistic
T = Float64
nobs = 1_000_000
y = rand(T, nobs)
pred = rand(T, 1, nobs)
K = 1
δ𝑤 = zeros(T, 2 * K + 1, nobs)
w = ones(T, nobs)
δ𝑤[end, :] .= w

# nthreads: 12
Threads.nthreads()

function update_grads_v1!(::Type{EvoTrees.Linear}, δ𝑤::Matrix, p::Matrix, y::Vector; kwargs...)
    @inbounds for i in eachindex(y)
        δ𝑤[1, i] = 2 * (p[1, i] - y[i]) * δ𝑤[3, i]
        δ𝑤[2, i] = 2 * δ𝑤[3, i]
    end
end
# 958.670 μs (0 allocations: 0 bytes)
@btime update_grads_v1!(L, δ𝑤, pred, y)

function update_grads_v2!(::Type{EvoTrees.Linear}, δ𝑤::Matrix, p::Matrix, y::Vector; kwargs...)
    @threads for i in eachindex(y)
        @inbounds δ𝑤[1, i] = 2 * (p[1, i] - y[i]) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = 2 * δ𝑤[3, i]
    end
end
# 958.670 μs (0 allocations: 0 bytes)
@btime update_grads_v2!(L, δ𝑤, pred, y)
