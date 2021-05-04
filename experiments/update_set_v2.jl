using Statistics
using StatsBase:sample
using Base.Threads:@threads
using BenchmarkTools
using Revise
using EvoTrees

n_obs = Int(1e6)
n_vars = 100
n_bins = 255
𝑖 = collect(1:n_obs);
𝑗 = collect(1:n_vars);
X_bin = rand(UInt8, n_obs, n_vars);

function split_set_chunk!(left, right, block, bid, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights, bsizes)
    left_count = 0
    right_count = 0
    @inbounds for i in eachindex(block)
        @inbounds if X_bin[block[i], feat] <= cond_bin
            left_count += 1
            left[offset + chunk_size * (bid - 1) + left_count] = block[i]
        else
            right_count += 1
            right[offset + chunk_size * (bid - 1) + right_count] = block[i]
            # right[offset + chunk_size * (bid - 1) + length(block) - right_count] = block[i]
            # right_count += 1
        end
    end
    lefts[bid] = left_count
    rights[bid] = right_count
    bsizes[bid] = length(block)
    return nothing
end

function split_set_threads!(out, left, right, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset, chunk_size=2^14) where {S}    

    left_count = 0 
    right_count = 0
    iter = Iterators.partition(𝑖, chunk_size)
    nblocks = length(iter)
    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)
    bsizes = zeros(Int, nblocks)

    @sync for (bid, block) in enumerate(iter)
        Threads.@spawn split_set_chunk!(left, right, block, bid, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights, bsizes)
    end

    left_sum = sum(lefts)
    left_cum = 0
    right_cum = 0
    @inbounds for bid in 1:nblocks
        view(out, offset + left_cum + 1:offset + left_cum + lefts[bid]) .= view(left, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + lefts[bid])
        view(out, offset + left_sum + right_cum + 1:offset + left_sum + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        left_cum += lefts[bid]
        right_cum += rights[bid]
    end
    return (view(out, offset + 1:offset + sum(lefts)), view(out, offset + sum(lefts)+1:offset + length(𝑖)))
end

iter = Iterators.partition(rand(5), 3)
for i in enumerate(iter)
    println(i)
end

n = Int(1e6)
nvars = 100
nbins = 64
𝑖 = collect(1:n);
𝑗 = collect(1:nvars);
X_bin = reshape(sample(UInt8.(1:nbins), n * nvars), n, nvars);
𝑖 = sample(𝑖, Int(5e5), replace=false, ordered=true);
child_bool = zeros(Bool, length(𝑖));
left = similar(𝑖)
right = similar(𝑖)
out = similar(𝑖)

offset = 0
feat = 15
cond_bin = 32
@time l, r = split_set_threads!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset, 2^14);
@btime split_set_threads!($out, $left, $right, $𝑖, $X_bin, $feat, $cond_bin, $offset, 2^14);
@code_warntype split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)

offset = 0
feat = 15
cond_bin = 32
lid1, rid1 = split_set_threads!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset)
offset = 0
feat = 14
cond_bin = 12
lid2, rid2 = split_set_threads!(out, left, right, lid1, X_bin, feat, cond_bin, offset)
offset = + length(lid1)
feat = 14
cond_bin = 12
lid3, rid3 = split_set_threads!(out, left, right, rid1, X_bin, feat, cond_bin, offset)

lid1_ = deepcopy(lid1)



