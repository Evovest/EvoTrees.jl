using Statistics
using StatsBase: sample
using Base.Threads: @threads
using BenchmarkTools
using Revise
using EvoTrees

n_obs = Int(1e6)
n_vars = 100
n_bins = 255
𝑖 = collect(1:n_obs)
𝑗 = collect(1:n_vars)
δ = rand(n_obs)
δ² = rand(n_obs)

hist_δ = zeros(n_bins, n_vars)
hist_δ² = zeros(n_bins, n_vars)
X_bin = rand(UInt8, n_obs, n_vars)

function iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
        end
    end
end

@time iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)

# takeaway : significant speedup from depth 3 if building all hit simultaneously
𝑖_4 = sample(𝑖, Int(n_obs/4), ordered=true)
@btime iter_1($X_bin, $hist_δ, $δ, $𝑖, $𝑗)
@btime iter_1($X_bin, $hist_δ, $δ, $𝑖_4, $𝑗)


# try adding all info on single array rather than seperate vectors
function iter_1B(X_bin, hist_δ, hist_δ², δ, δ², 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            @inbounds hist_δ[X_bin[i,j], j] += δ[i]
            @inbounds hist_δ²[X_bin[i,j], j] += δ²[i]
        end
    end
end

@btime iter_1B($X_bin, $hist_δ, $hist_δ², $δ, $δ², $𝑖, $𝑗)

# try adding all info on single array rather than seperate vectors
δ2 = rand(2, n_obs)
hist_δ2 = zeros(n_bins, 2, n_vars)
function iter_2(X_bin, hist_δ2, δ2, 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            # view(hist_δ2, X_bin[i,j], j, :) .+= view(δ2, i, :)
            @inbounds for k in 1:2
                hist_δ2[X_bin[i,j], k, j] += δ2[k, i]
                # @inbounds hist_δ2[X_bin[i,j], 1, j] += δ2[i, 1]
                # @inbounds hist_δ2[X_bin[i,j], 2, j] += δ2[i, 2]
            end
        end
    end
end
@time iter_2(X_bin, hist_δ2, δ2, 𝑖_4, 𝑗)
@btime iter_2($X_bin, $hist_δ2, $δ2, $𝑖, $𝑗)


# integrate a leaf id
hist_δ2 = zeros(n_bins, 2, n_vars, 8);
@time hist_δ2 .= 0;
@time hist_δ2 .* 0;
function iter_3(X_bin, hist, δ2, 𝑖, 𝑗, leaf)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            @inbounds for k in 1:2
                # view(hist_δ2, X_bin[i,j], j, :) .+= view(δ2, i, :)
                @inbounds hist[X_bin[i,j], k, j, leaf[i]] += δ2[k, i]
            end
        end
    end
end

leaf_vec = ones(UInt8, n_obs)
@time iter_3(X_bin, hist_δ2, δ2, 𝑖_4, 𝑗, leaf_vec);
@btime iter_3($X_bin, $hist_δ2, $δ2, $𝑖, $𝑗, $leaf_vec);
