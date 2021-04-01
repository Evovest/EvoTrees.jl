# using Statistics
using StatsBase:sample
# using Base.Threads:@threads
using BenchmarkTools

n_obs = Int(1e6)
n_vars = 100
n_bins = 64
K = 3
𝑖 = collect(1:n_obs)
δ = rand(n_obs, K)
hist = zeros(K, n_bins, n_vars);
X_bin = sample(UInt8.(1:n_bins), n_obs * n_vars);
X_bin = reshape(X_bin, n_obs, n_vars);

function iter_1(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds for i in 𝑖
        @inbounds for k in 1:3
            hist[k, X_bin[i,1], 1] += δ[i,k]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_1(X_bin, hist, δ, 𝑖_sample)
@btime iter_1($X_bin, $hist, $δ, $𝑖_sample)



function iter_2(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds @simd for i in CartesianIndices(𝑖)
        @inbounds @simd for k in 1:3
            hist[k, X_bin[𝑖[i],1], 1] += δ[𝑖[i],k]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_2(X_bin, hist, δ, 𝑖_sample)
@btime iter_2($X_bin, $hist, $δ, $𝑖_sample)



# slower
δ = rand(K, n_obs)
hist = zeros(K, n_bins, n_vars);
X_bin = sample(UInt8.(1:n_bins), n_obs * n_vars);
X_bin = reshape(X_bin, n_obs, n_vars);

function iter_1(X_bin, hist, δ, 𝑖)
    hist .= 0.0
    @inbounds for i in 𝑖
        @inbounds for k in 1:3
            hist[k, X_bin[i,1], 1] += δ[k,i]
        end
    end
end

𝑖_sample = sample(𝑖, Int(n_obs / 2), ordered=true)

@time iter_1(X_bin, hist, δ, 𝑖_sample)
@btime iter_1($X_bin, $hist, $δ, $𝑖_sample)
