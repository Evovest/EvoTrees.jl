using Statistics
using StatsBase:sample
using Base.Threads:@threads
using BenchmarkTools

n_obs = Int(1e6)
n_vars = 100
n_bins = 64
𝑖 = collect(1:n_obs)
δ = rand(n_obs)

hist = zeros(n_bins, n_vars);
X_bin = sample(UInt8.(1:n_bins), n_obs * n_vars);
X_bin = reshape(X_bin, n_obs, n_vars);

function iter_1(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds for i in 𝑖
        hist[X_bin[i,1], 1] += δ[i]
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_1(X_bin, hist, δ, 𝑖_sample)
@btime iter_1($X_bin, $hist, $δ, $𝑖_sample)
