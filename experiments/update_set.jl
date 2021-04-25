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
δ = rand(n_obs);
δ² = rand(n_obs);

hist_δ = zeros(n_bins, n_vars);
hist_δ² = zeros(n_bins, n_vars);
X_bin = rand(UInt8, n_obs, n_vars);

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


# split row ids into left and right based on best split condition
function split_set_1!(left::V, right::V, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset) where {S,V}
    
    left_count = 0 
    right_count = 0

    @inbounds for i in 1:length(𝑖)
        @inbounds if X_bin[𝑖[i], feat] <= cond_bin
            left_count += 1
            left[offset + left_count] = 𝑖[i]
        else
            right[offset + length(𝑖) - right_count] = 𝑖[i]
            right_count += 1
        end
    end
    # return (left[1:left_count], right[1:right_count])
    return (view(left, (offset + 1):(offset + left_count)), view(right, (offset + length(𝑖)):-1:(offset + left_count + 1)))
    # return nothing
end

n = Int(1e6)
nvars = 100
nbins = 64
𝑖 = collect(1:n)
𝑗 = collect(1:nvars)
X_bin = reshape(sample(UInt8.(1:nbins), n * nvars), n, nvars)
left = similar(𝑖)
right = similar(𝑖)

𝑖 = sample(𝑖, Int(5e5), replace=false, ordered=true)

offset = 0
feat = 15
cond_bin = 32
@time lid2, rid2 = split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)
@btime split_set_1!($left, $right, $𝑖, $X_bin, $feat, $cond_bin, $offset)
@code_warntype split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)

offset = 0
feat = 15
cond_bin = 32
lid1, rid1 = split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)
offset = 0
feat = 14
cond_bin = 12
lid2, rid2 = split_set_1!(left, right, lid1, X_bin, feat, cond_bin, offset)
offset = + length(lid1)
feat = 14
cond_bin = 12
lid3, rid3 = split_set_1!(left, right, rid1, X_bin, feat, cond_bin, offset)

lid1_ = deepcopy(lid1)

𝑖
unique(vcat(lid1, rid1))
unique(vcat(lid1))
unique(vcat(rid1))
unique(sort(vcat(lid2, rid2)))
unique(sort(vcat(lid3, rid3)))
unique(sort(vcat(lid2, rid2, lid3, rid3)))

# split row ids into left and right based on best split condition
function split_set_2!(left, right, 𝑖, x_bin, feat, cond_bin)
    
    left_count = 0 
    right_count = 0

    @inbounds for i in 1:length(𝑖)
        if x_bin[i] <= cond_bin
            left_count += 1
            left[left_count] = 𝑖[i]
        else
            right_count += 1
            right[right_count] = 𝑖[i]
        end
    end
    # return (left[1:left_count], right[1:right_count])
    return (view(left, 1:left_count), view(right, 1:right_count))
    # return nothing
end

n = Int(1e6)
nvars = 100
nbins = 64
𝑖 = collect(1:n)
𝑗 = collect(1:nvars)
X_bin = reshape(sample(UInt8.(1:nbins), n * nvars), n, nvars)
left = similar(𝑖)
right = similar(𝑖)

feat = 15
cond_bin = 32
@time left, right = split_set_2!(left, right, 𝑖, X_bin[:,feat], feat, cond_bin)
@btime split_set_2!($left, $right, $𝑖, $X_bin[:,feat], $feat, $cond_bin)
@btime split_set_2!($left, $right, $𝑖, $view(X_bin, :, feat), $feat, $cond_bin)



# function split_set_bool!(child_bool::AbstractVector{Bool}, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset) where {S}    
#     left_count = 0 
#     right_count = 0
#     @inbounds for i in eachindex(𝑖)
#         child_bool[i + offset] = X_bin[𝑖[i], feat] <= cond_bin
#     end
#     # return (view(𝑖, child_bool[offset + 1:offset + length(𝑖)]), view(𝑖, .!child_bool[offset + 1:offset + length(𝑖)]))
#     # return (view(𝑖, view(child_bool, (offset + 1):(offset + length(𝑖)))), view(𝑖, view(child_bool, (offset + 1):(offset + length(𝑖)))))
#     # return (view(𝑖, child_bool), view(𝑖, child_bool))
#     return view(𝑖, child_bool)
# end

function split_set_chunk!(left, right, block, bid, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights, bsizes)
    left_count = 0
    right_count = 0
    @inbounds for i in eachindex(block)
        @inbounds if X_bin[block[i], feat] <= cond_bin
            left_count += 1
            left[offset + chunk_size * (bid - 1) + left_count] = block[i]
        else
            right[offset + chunk_size * (bid - 1) + length(block) - right_count] = block[i]
            right_count += 1
        end
    end
    lefts[bid] = left_count
    rights[bid] = right_count
    bsizes[bid] = length(block)
    return nothing
end

function split_set_threads!(left, right, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset, chunk_size=2^14) where {S}    

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

    left_cum = 0
    @inbounds for bid in 1:nblocks
        view(left, offset + left_cum + 1:offset + left_cum + lefts[bid]) .= view(left, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + lefts[bid])
        # view(right, offset + right_cum + 1:offset + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        # view(right, offset + length(𝑖) - right_cum:-1:offset + length(𝑖) - right_cum - rights[bid] + 1) .= view(right, offset + chunk_size * (bid - 1) + bsizes[bid]:-1:offset + chunk_size * (bid - 1) + lefts[bid]+1)
        left_cum += lefts[bid]
    end
    
    right_cum = 0
    @inbounds for bid in nblocks:-1:1
        # view(right, offset + right_cum + 1:offset + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        view(right, offset + length(𝑖) - right_cum:-1:offset + length(𝑖) - right_cum - rights[bid] + 1) .= view(right, offset + chunk_size * (bid - 1) + lefts[bid] + 1:offset + chunk_size * (bid - 1) + bsizes[bid])
        right_cum += rights[bid]
    end

    return (view(left, offset + 1:offset + sum(lefts)), view(right, offset + sum(lefts) + 1:offset + length(𝑖)))
    # return (view(left, offset + 1:offset + sum(lefts)), view(right, offset + 1:offset + sum(rights)))
    # return (left[offset + 1:offset + sum(lefts)], right[offset + 1:offset + sum(rights)])
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

offset = 0
feat = 15
cond_bin = 32
@time l2, r2 = split_set_threads!(left, right, 𝑖, X_bin, feat, cond_bin, offset, 2^14);
@btime split_set_threads!($left, $right, $𝑖, $X_bin, $feat, $cond_bin, $offset, 2^14);
@code_warntype split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)

offset = 0
feat = 15
cond_bin = 32
lid1, rid1 = split_set_1!(left, right, 𝑖, X_bin, feat, cond_bin, offset)
offset = 0
feat = 14
cond_bin = 12
lid2, rid2 = split_set_1!(left, right, lid1, X_bin, feat, cond_bin, offset)
offset = + length(lid1)
feat = 14
cond_bin = 12
lid3, rid3 = split_set_1!(left, right, rid1, X_bin, feat, cond_bin, offset)

lid1_ = deepcopy(lid1)
