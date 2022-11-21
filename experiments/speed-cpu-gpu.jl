using Revise
using Statistics
using StatsBase: sample, sample!
using EvoTrees
using BenchmarkTools
using CUDA
using Base.Threads: nthreads, threadid, @threads, @spawn
using Random: seed!, Xoshiro, MersenneTwister, TaskLocalRNG

# prepare a dataset
features = rand(Int(1.25e6), 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X, 1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

x_train, x_eval = X[𝑖_train, :], X[𝑖_eval, :]
y_train, y_eval = Y[𝑖_train], Y[𝑖_eval]

###########################
# Tree CPU
###########################
params_c = EvoTreeRegressor(
    T = Float32,
    loss = :linear,
    nrounds = 100,
    lambda = 0.1,
    gamma = 0.0,
    eta = 0.1,
    max_depth = 6,
    min_weight = 1.0,
    rowsample = 0.5,
    colsample = 0.5,
    nbins = 64,
);

model_c, cache_c = EvoTrees.init_evotree(params_c; x_train, y_train);

# initialize from cache
X_size = size(cache_c.x_bin)

# 897.800 μs (6 allocations: 736 bytes)
@time EvoTrees.update_grads!(cache_c.∇, cache_c.pred, cache_c.y, params_c)
# @btime EvoTrees.update_grads!($params_c.loss, $cache_c.δ𝑤, $cache_c.pred_cpu, $cache_c.Y_cpu, $params_c.α)

# select random rows and cols
cache_c.nodes[1].is = EvoTrees.subsample(cache_c.is, cache_c.mask, params_c.rowsample, params_c.rng);
length(cache_c.nodes[1].is)
sample!(params_c.rng, cache_c.js_, cache_c.js, replace = false, ordered = true);
# @btime sample!(params_c.rng, cache_c.𝑖_, cache_c.nodes[1].𝑖, replace=false, ordered=true);

is = cache_c.nodes[1].is
js = cache_c.js

L = EvoTrees.Linear
K = 1
T = Float32

######################################################
# sampling experiements
######################################################
# 7.892 ms (0 allocations: 0 bytes)
function get_rand!(rng, mask)
    @threads for i in eachindex(mask)
        @inbounds mask[i] = rand(rng, UInt8)
    end
end
function get_rand_repro_A!(rngs, mask)
    @threads for i in eachindex(mask)
        tid = threadid()
        @inbounds mask[i] = rand(rngs[tid], UInt8)
    end
end
function get_rand_repro_B!(rngs, mask)
    nblocks = length(rngs)
    chunk_size = cld(length(mask), nblocks)
    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1) + 1
        i_stop = min(length(mask), i_start + chunk_size - 1)
        rng = rngs[bid]
        @inbounds for i = i_start:i_stop
            @inbounds mask[i] = rand(rng, UInt8)
        end
    end
end
function get_rand_repro_C!(rngs, mask)
    nblocks = length(rngs)
    chunk_size = cld(length(mask), nblocks)
    @threads for bid = 1:nblocks
        tx = threadid()
        i_start = chunk_size * (tx - 1) + 1
        i_stop = min(length(mask), i_start + chunk_size - 1)
        rng = rngs[tx]
        @inbounds for i = i_start:i_stop
            mask[i] = rand(rng, UInt8)
        end
    end
end

nobs = 1_000_000

mask_ori = zeros(UInt32, nobs)
rng = TaskLocalRNG()
seed!(rng, 123)
# rng = Xoshiro(123 + 1)
# rng = MersenneTwister(123 + 1)
get_rand!(rng, mask_ori)
@btime get_rand!($rng, $mask_ori)

mask_A = zeros(UInt32, nobs)
rngs = [Random.MersenneTwister(123 + i) for i = 1:nthreads()]
get_rand_repro_A!(rngs, mask_A)
@btime get_rand_repro_A!($rngs, $mask_A)

mask_B = zeros(UInt32, nobs)
rngs = [Xoshiro(123 + i) for i = 1:nthreads()]
get_rand_repro_B!(rngs, mask_B)
@btime get_rand_repro_B!($rngs, $mask_B)

mask_C = zeros(UInt32, nobs)
rngs = [Random.MersenneTwister(123 + i) for i = 1:nthreads()]
get_rand_repro_C!(rngs, mask_C)
@btime get_rand_repro_C!($rngs, $mask_C)

rngs = [Random.MersenneTwister(123 + i) for i = 1:nthreads()]
rng1 = Random.MersenneTwister(123)
rng2 = Random.MersenneTwister(124)

rand(rngs[1], UInt32)
rand(rng1, UInt32)
rand(rng2, UInt32)

@btime sample!(
    $params_c.rng,
    $cache_c.𝑖_,
    $cache_c.nodes[1].𝑖,
    replace = false,
    ordered = true,
);

# 21.684 ms (2 allocations: 7.63 MiB)
@btime sample!(
    $params_c.rng,
    $cache_c.𝑖_,
    $cache_c.nodes[1].𝑖,
    replace = false,
    ordered = false,
);

# 27.441 ms (0 allocations: 0 bytes)
@btime sample!(
    $params_c.rng,
    $cache_c.𝑖_,
    $cache_c.nodes[1].𝑖,
    replace = true,
    ordered = true,
);

src = zeros(Bool, length(cache_c.𝑖_))
target = zeros(Bool, length(cache_c.𝑖_))

# 58.000 μs (3 allocations: 976.69 KiB)
@btime rand(Bool, length(src));
# 1.452 ms (3 allocations: 7.63 MiB)
@btime rand(Float64, length(src));
# 507.800 μs (3 allocations: 3.81 MiB)
@btime rand(Float32, length(src));
@btime rand(Float16, length(src));
# 500.000 μs (3 allocations: 3.81 MiB)
@btime rand(UInt32, length(src));
# 244.800 μs (3 allocations: 1.91 MiB)
@btime rand(UInt16, length(src));
# 62.000 μs (3 allocations: 976.69 KiB)
@btime rand(UInt8, length(src));

function get_rand!(mask)
    @threads for i in eachindex(mask)
        @inbounds mask[i] = rand(UInt8)
    end
end
mask = zeros(UInt8, length(cache_c.𝑖_))
# 126.100 μs (48 allocations: 5.08 KiB)
@btime get_rand!($mask)
function subsample_kernelA!(mask, cond, out_view)
    count = 0
    @inbounds for i in eachindex(out_view)
        if mask[i] <= cond
            count += 1
            out_view[count] = i
        end
    end
    return count
end
function subsampleA(out, mask, rowsample)

    get_rand!(mask)
    cond = round(UInt8, 255 * rowsample)
    nblocks = ceil(Int, min(length(out) / 100_000, Threads.nthreads()))
    chunk_size = ceil(Int, length(out) / nblocks)
    counts = zeros(Int, nblocks)

    @threads for bid in eachindex(counts)
        i_start = chunk_size * (bid - 1) + 1
        i_stop = (bid == nblocks) ? length(out) : i_start + chunk_size - 1
        counts[bid] = subsample_kernelA!(mask, cond, view(out, i_start:i_stop))
    end

    count = 0
    @inbounds for bid in eachindex(counts)
        i_start = chunk_size * (bid - 1) + 1
        view(out, count+1:(count+counts[bid])) .= view(out, i_start:(i_start+counts[bid]-1))
        count += counts[bid]
    end
    return view(out, 1:count)
end
out = zeros(UInt32, 1_000_000)
mask = rand(UInt8, length(out))
@time out_view = subsampleA(out, mask, 0.5);
Int(minimum(out_view))
Int(maximum(out_view))
@btime subsampleA($out, $mask, 0.5);

function subsample(out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat)
    get_rand!(mask)
    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(out), min(length(out) ÷ 1024, Threads.nthreads()))
    nblocks = cld(length(out), chunk_size)
    counts = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1) + 1
        i_stop = bid == nblocks ? length(out) : i_start + chunk_size - 1
        count = 0
        i = i_start
        for i = i_start:i_stop
            if mask[i] <= cond
                out[i_start+count] = i
                count += 1
            end
        end
        counts[bid] = count
    end
    counts_cum = cumsum(counts) .- counts
    @threads for bid = 1:nblocks
        count_cum = counts_cum[bid]
        i_start = chunk_size * (bid - 1)
        @inbounds for i = 1:counts[bid]
            out[count_cum+i] = out[i_start+i]
        end
    end
    return view(out, 1:sum(counts))
end
out = zeros(UInt32, 1_000_000)
mask = rand(UInt8, length(out))
@time out_view = subsample(out, mask, 0.5);
Int(minimum(out_view))
Int(maximum(out_view))
@btime subsample($out, $mask, 0.5);
@code_warntype subsample(out, mask, 0.5);

function debug(n)
    for i = 1:n
        out = zeros(UInt32, 1_000_000)
        mask = rand(UInt8, length(out))
        out_view = subsample(out, mask, 0.5)
        min = Int(minimum(out_view))
        min_count = sum(out_view .== 0)
        if min == 0
            @info "$min_count 0s at iteration $i"
        end
    end
end
debug(100)


function get_rand_kernel!(mask)
    tix = threadIdx().x
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(mask)
    niter = cld(i_max, bdx * gdx)
    for iter = 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            mask[i] = rand(UInt8)
        end
    end
    sync_threads()
end
function get_rand_gpu!(mask)
    threads = (1024,)
    blocks = (256,)
    @cuda threads = threads blocks = blocks get_rand_kernel!(mask)
    CUDA.synchronize()
end

function subsample_step_1_kernel(out, mask, cond, counts, chunk_size)

    bid = blockIdx().x
    gdim = gridDim().x

    i_start = chunk_size * (bid - 1) + 1
    i_stop = bid == gdim ? length(out) : i_start + chunk_size - 1
    count = 0

    @inbounds for i = i_start:i_stop
        @inbounds if mask[i] <= cond
            out[i_start+count] = i
            count += 1
        end
    end
    sync_threads()
    @inbounds counts[bid] = count
    sync_threads()
end

function subsample_step_2_kernel(out, counts, counts_cum, chunk_size)
    bid = blockIdx().x
    count_cum = counts_cum[bid]
    i_start = chunk_size * (bid - 1)
    @inbounds for i = 1:counts[bid]
        out[count_cum+i] = out[i_start+i]
    end
    sync_threads()
end

function subsample_gpu(out::CuVector, mask::CuVector, rowsample::AbstractFloat)
    get_rand_gpu!(mask)
    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(out), min(length(out) ÷ 128, 2048))
    nblocks = cld(length(out), chunk_size)
    counts = CUDA.zeros(Int, nblocks)

    blocks = (nblocks,)
    threads = (1,)

    @cuda blocks = nblocks threads = 1 subsample_step_1_kernel(
        out,
        mask,
        cond,
        counts,
        chunk_size,
    )
    CUDA.synchronize()
    counts_cum = cumsum(counts) - counts
    @cuda blocks = nblocks threads = 1 subsample_step_2_kernel(
        out,
        counts,
        counts_cum,
        chunk_size,
    )
    CUDA.synchronize()
    return view(out, 1:sum(counts))
end

out = CUDA.zeros(UInt32, 1_000_000)
mask = CUDA.zeros(UInt8, length(out))
CUDA.@time get_rand_gpu!(mask)
# 39.100 μs (5 allocations: 304 bytes)
# @btime get_rand_gpu!(mask)

CUDA.@time out_view = subsample_gpu(out, mask, 0.5);
Int(minimum(out_view))
Int(maximum(out_view))
@btime subsample_gpu(out, mask, 1.0);

##########################################################
# end subsample tests
##########################################################

# 12.058 ms (2998 allocations: 284.89 KiB)
tree = EvoTrees.Tree{L,K,T}(params_c.max_depth)
@time EvoTrees.grow_tree!(
    tree,
    cache_c.nodes,
    params_c,
    cache_c.δ𝑤,
    cache_c.edges,
    cache_c.js,
    cache_c.left,
    cache_c.left,
    cache_c.right,
    cache_c.x_bin,
    cache_c.monotone_constraints,
)
@code_warntype EvoTrees.grow_tree!(
    tree,
    cache_c.nodes,
    params_c,
    cache_c.δ𝑤,
    cache_c.edges,
    cache_c.js,
    cache_c.left,
    cache_c.left,
    cache_c.right,
    cache_c.x_bin,
    cache_c.monotone_constraints,
)
@btime EvoTrees.grow_tree!(
    $EvoTrees.Tree{L,K,T}(params_c.max_depth),
    $cache_c.nodes,
    $params_c,
    $cache_c.δ𝑤,
    $cache_c.edges,
    $cache_c.js,
    $cache_c.left,
    $cache_c.left,
    $cache_c.right,
    $cache_c.x_bin,
    $cache_c.monotone_constraints,
)

# push!(model_c.trees, tree)
# 993.447 μs (75 allocations: 7.17 KiB)
@btime EvoTrees.predict!(cache_c.pred, tree, cache_c.x)

δ𝑤, edges, x_bin, nodes, out, left, right, mask, monotone_constraints = cache_c.δ𝑤,
cache_c.edges,
cache_c.x_bin,
cache_c.nodes,
cache_c.out,
cache_c.left,
cache_c.right,
cache_c.mask,
cache_c.monotone_constraints;

# Float32: 2.984 ms (73 allocations: 6.52 KiB)
# Float64: 5.020 ms (73 allocations: 6.52 KiB)

𝑖 = cache_c.nodes[1].𝑖
# mask = rand(Bool, length(𝑖))
@time EvoTrees.update_hist!(L, nodes[1].h, δ𝑤, x_bin, 𝑖, 𝑗)
@btime EvoTrees.update_hist!($L, $nodes[1].h, $δ𝑤, $x_bin, $𝑖, $𝑗)
@code_warntype EvoTrees.update_hist!(L, nodes[1].h, δ𝑤, x_bin, 𝑖, 𝑗)

j = 1
n = 1
nodes[1].∑ .= vec(sum(δ𝑤[:, 𝑖], dims = 2))
@time EvoTrees.update_gains!(nodes[n], 𝑗, params_c, K, monotone_constraints)
# 12.160 μs (0 allocations: 0 bytes)
@btime EvoTrees.update_gains!($nodes[n], $𝑗, $params_c, $K, $monotone_constraints)
nodes[1].gains;
# @code_warntype EvoTrees.update_gains!(nodes[n], 𝑗, params_c, K, monotone_constraints)

# 5.793 μs (1 allocation: 32 bytes)
best = findmax(nodes[n].gains)
@btime best = findmax(nodes[n].gains)

tree.cond_bin[n] = best[2][1]
tree.feat[n] = best[2][2]

Int.(tree.cond_bin[n])
# tree.cond_bin[n] = 32

# 204.900 μs (1 allocation: 96 bytes)
offset = 0
@time EvoTrees.split_set!(left, right, 𝑖, X_bin, tree.feat[n], tree.cond_bin[n], offset)
@btime EvoTrees.split_set!(
    $left,
    $right,
    $𝑖,
    $X_bin,
    $tree.feat[n],
    $tree.cond_bin[n],
    $offset,
)
@code_warntype EvoTrees.split_set!(left, right, 𝑖, X_bin, tree.feat[n], tree.cond_bin[n])

# 186.294 μs (151 allocations: 15.06 KiB)
@time _left, _right = EvoTrees.split_set_threads!(
    out,
    left,
    right,
    𝑖,
    x_bin,
    tree.feat[n],
    tree.cond_bin[n],
    offset,
);
@btime EvoTrees.split_set_threads!(
    $out,
    $left,
    $right,
    $𝑖,
    $x_bin,
    $tree.feat[n],
    $tree.cond_bin[n],
    $offset,
);

###################################################
# GPU
###################################################
params_g = EvoTreeRegressor(
    T = Float32,
    loss = :linear,
    nrounds = 100,
    lambda = 1.0,
    gamma = 0.1,
    eta = 0.1,
    max_depth = 6,
    min_weight = 1.0,
    rowsample = 0.5,
    colsample = 0.5,
    nbins = 64,
);

model_g, cache_g = EvoTrees.init_evotree_gpu(params_g; x_train, y_train);

x_size = size(cache_g.x_bin);

# select random rows and cols
𝑖c = cache_g.𝑖_[sample(
    params_g.rng,
    cache_g.𝑖_,
    ceil(Int, params_g.rowsample * x_size[1]),
    replace = false,
    ordered = true,
)]
𝑖 = CuVector(𝑖c)
𝑗c = cache_g.𝑗_[sample(
    params_g.rng,
    cache_g.𝑗_,
    ceil(Int, params_g.colsample * x_size[2]),
    replace = false,
    ordered = true,
)]
𝑗 = CuVector(𝑗c)
cache_g.nodes[1].𝑖 = 𝑖
cache_g.𝑗 .= 𝑗c

L = EvoTrees.Linear
K = 1
T = Float32
# build a new tree
# 144.600 μs (23 allocations: 896 bytes) - 5-6 X time faster on GPU
@time CUDA.@sync EvoTrees.update_grads_gpu!(cache_g.δ𝑤, cache_g.pred, cache_g.y, params_g)
# @btime CUDA.@sync EvoTrees.update_grads_gpu!($params_g.loss, $cache_g.δ𝑤, $cache_g.pred_gpu, $cache_g.Y_gpu)
# sum Gradients of each of the K parameters and bring to CPU

# 33.447 ms (6813 allocations: 307.27 KiB)
tree = EvoTrees.TreeGPU{L,K,T}(params_g.max_depth)
sum(cache_g.δ𝑤[:, cache_g.nodes[1].𝑖], dims = 2)
CUDA.@time EvoTrees.grow_tree_gpu!(
    tree,
    cache_g.nodes,
    params_g,
    cache_g.δ𝑤,
    cache_g.edges,
    CuVector(cache_g.𝑗),
    cache_g.out,
    cache_g.left,
    cache_g.right,
    cache_g.x_bin,
    cache_g.monotone_constraints,
)
@btime EvoTrees.grow_tree_gpu!(
    EvoTrees.TreeGPU{L,K,T}(params_g.max_depth),
    cache_g.nodes,
    params_g,
    $cache_g.δ𝑤,
    $cache_g.edges,
    $𝑗,
    $cache_g.out,
    $cache_g.left,
    $cache_g.right,
    $cache_g.x_bin,
    $cache_g.monotone_constraints,
);
@code_warntype EvoTrees.grow_tree_gpu!(
    EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ),
    params_g,
    cache_g.δ,
    cache_g.hist,
    cache_g.histL,
    cache_g.histR,
    cache_g.gains,
    cache_g.edges,
    𝑖,
    𝑗,
    𝑛,
    cache_g.x_bin,
);

push!(model_g.trees, tree);
# 2.736 ms (93 allocations: 13.98 KiB)
@time CUDA.@sync EvoTrees.predict!(cache_g.pred_gpu, tree, cache_g.X_bin)
@btime CUDA.@sync EvoTrees.predict!($cache_g.pred_gpu, $tree, $cache_g.X_bin)

###########################
# Tree GPU
###########################
δ𝑤, edges, x_bin, nodes, out, left, right = cache_g.δ𝑤,
cache_g.edges,
cache_g.x_bin,
cache_g.nodes,
cache_g.out,
cache_g.left,
cache_g.right;

# 2.571 ms (1408 allocations: 22.11 KiB)
# 𝑗2 = CuArray(sample(UInt32.(1:100), 50, replace=false, ordered=true))
@time EvoTrees.update_hist_gpu!(nodes[1].h, δ𝑤, x_bin, 𝑖, 𝑗)
@btime EvoTrees.update_hist_gpu!($nodes[1].h, $δ𝑤, $x_bin, $𝑖, $𝑗)
# @code_warntype EvoTrees.update_hist_gpu!(hist, δ, X_bin, 𝑖, 𝑗, 𝑛)


# 72.100 μs (186 allocations: 6.00 KiB)
n = 1
nodes[1].∑ .= vec(sum(δ𝑤[:, 𝑖], dims = 2))
CUDA.@time EvoTrees.update_gains_gpu!(params_g.loss, nodes[n], 𝑗, params_g, K)
@btime EvoTrees.update_gains_gpu!($params_g.loss, $nodes[n], $𝑗, $params_g, $K)

tree = EvoTrees.TreeGPU(params_g.max_depth, model_g.K, params_g.λ)
best = findmax(nodes[n].gains)
if best[2][1] != params_g.nbins && best[1] > nodes[n].gain + params_g.γ
    tree.gain[n] = best[1]
    tree.cond_bin[n] = best[2][1]
    tree.feat[n] = best[2][2]
    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
end

tree.split[n] = tree.cond_bin[n] != 0
tree.feat[n]
Int(tree.cond_bin[n])

# 673.900 μs (600 allocations: 29.39 KiB)
offset = 0
_left, _right = EvoTrees.split_set_threads_gpu!(
    out,
    left,
    right,
    𝑖,
    X_bin,
    tree.feat[n],
    tree.cond_bin[n],
    offset,
)
@time EvoTrees.split_set_threads_gpu!(
    out,
    left,
    right,
    𝑖,
    X_bin,
    tree.feat[n],
    tree.cond_bin[n],
    offset,
)
@btime EvoTrees.split_set_threads_gpu!(
    $out,
    $left,
    $right,
    $𝑖,
    $X_bin,
    $tree.feat[n],
    $tree.cond_bin[n],
    $offset,
)
