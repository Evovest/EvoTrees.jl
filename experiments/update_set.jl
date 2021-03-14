using Statistics
using StatsBase: sample
using Base.Threads: @threads
using BenchmarkTools
using Revise
using EvoTrees
using CUDA

n_obs = Int(1e6)
n_vars = 100
n_bins = 255
𝑖 = collect(1:n_obs);
𝑗 = collect(1:n_vars);
δ = rand(n_obs);
δ² = rand(n_obs);

hist_δ = zeros(n_bins, n_vars);
hist_δ² = zeros(n_bins, n_vars);
X_bin = rand(UInt8, n_obs, n_vars);
X_bin_g = CuArray(X_bin)
𝑖g = CuArray(𝑖)

@time EvoTrees.update_set(𝑖, 4, view(X_bin, :, 1));
@btime EvoTrees.update_set(𝑖, 4, view(X_bin, :, 1));
@btime EvoTrees.update_set(𝑖, 4, X_bin[:, 1]);

@btime EvoTrees.update_set_gpu(𝑖g, 32, view(X_bin_g, :, 1));
@btime EvoTrees.update_set_gpu(𝑖g, 32, X_bin_g[:, 1]);
CUDA.@time EvoTrees.update_set_gpu(𝑖g, 4, view(X_bin_g, :, 1));
CUDA.@time EvoTrees.update_set_gpu(𝑖g, 4, X_bin_g[:, 1]);

left, right = EvoTrees.update_set(𝑖, 4, view(X_bin, :, 1))
left, right = EvoTrees.update_set_gpu(𝑖g, 4, view(X_bin_g, :, 1))

# split row ids into left and right based on best split condition
function update_set_1(set, best, x_bin)
    left = similar(set)
    right = similar(set)
    left_count = 0
    right_count = 0
    @inbounds for i in set
        if x_bin[i] <= best
            left_count += 1
            left[left_count] = i
        else
            right_count += 1
            right[right_count] = i
        end
    end
    resize!(left, left_count)
    resize!(right, right_count)
    return left, right
end

@time update_set_1(𝑖, 16, X_bin[:,1]);
@btime update_set_1($𝑖, 16, $X_bin[:,1]);
@btime update_set_1($𝑖, 64, $X_bin[:,1]);
@btime update_set_1($𝑖, 128, $X_bin[:,1]);
@btime update_set_1($𝑖, 240, $X_bin[:,1]);

# add a leaf id update - to indicate to which leaf the set is associated
function update_set_2!(leaf_vec::Vector{T}, set, best_feat, best_cond, x_bin, depth::T) where {T}
    @inbounds for i in set
        left_id = leaf_vec[i] + 2^depth
        right_id = left_id + 1
        x_bin[i, best_feat[leaf_vec[i]]] <= best_cond[leaf_vec[i]] ? leaf_vec[i] = left_id : leaf_vec[i] = right_id
    end
end

leaf_vec = ones(UInt16, n_obs);
leaf_id = 0
depth = UInt16(1)
depth = 1
best_feat = UInt16.(sample(1:100, 100000))
best_cond = rand(UInt16, 100000);

@time update_set_2!(leaf_vec, 𝑖, best_feat, best_cond, X_bin, depth);
@btime update_set_2!($leaf_vec, $𝑖, $best_feat, $best_cond, $X_bin, $depth);
Int.(leaf_vec)
