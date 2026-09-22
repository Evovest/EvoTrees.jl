using KernelAbstractions
using Atomix

"""
    partition_flag_kernel!(flag, is, nidx, x_bin, cond_feats, cond_bins, feattypes)

For each position `p` of `is`, `flag[p] = 1` if the row goes to the left child of its node's split
and `0` otherwise. Rows of nodes that were not split get `0`; [`partition_scatter_kernel!`](@ref)
leaves them in place.
"""
@kernel function partition_flag_kernel!(
    flag,
    @Const(is),
    @Const(nidx),
    @Const(x_bin),
    @Const(cond_feats),
    @Const(cond_bins),
    @Const(feattypes),
)
    p = @index(Global)
    @inbounds if p <= length(is)
        obs = is[p]
        node = nidx[obs]
        bin = cond_bins[node]
        is_left = false
        if bin != 0
            feat = cond_feats[node]
            is_left = feattypes[feat] ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
        end
        flag[p] = UInt32(is_left)
    end
end

"""
    partition_scatter_kernel!(dst, nidx, node_off, node_cnt, is, flag, scan, cond_bins)

Stable split of each split node's range of `is` into its left rows then its right rows, written
to `dst`, and move each row to its child in `nidx`. Rows of unsplit nodes are copied in place.
`scan[p]` is the number of left rows before position `p`, so a node's range `(o, o + c]` holds
`scan[o + c + 1] - scan[o + 1]` left rows. The thread of a node's first row records the row
ranges of its two children in `node_off` and `node_cnt`.
"""
@kernel function partition_scatter_kernel!(
    dst,
    nidx::AbstractVector{T},
    node_off,
    node_cnt,
    @Const(is),
    @Const(flag),
    @Const(scan),
    @Const(cond_bins),
) where {T<:Unsigned}
    p = @index(Global)
    @inbounds if p <= length(is)
        obs = is[p]
        node = nidx[obs]
        if cond_bins[node] == 0
            dst[p] = obs
        else
            o = Int(node_off[node])
            c = Int(node_cnt[node])
            nleft = Int(scan[o+c+1]) - Int(scan[o+1])
            lefts_before = Int(scan[p]) - Int(scan[o+1])
            child = node << 1
            if p == o + 1
                node_off[child], node_cnt[child] = o, nleft
                node_off[child+1], node_cnt[child+1] = o + nleft, c - nleft
            end
            if flag[p] != 0
                dst[o+lefts_before+1] = obs
            else
                dst[o+nleft+(p-1-o-lefts_before)+1] = obs
                child += one(T)
            end
            nidx[obs] = child
        end
    end
end

"""
    partition_rows!(dst, is, cache, backend)

Write `is` to `dst` reordered after the splits of a depth, so that each node's rows are
contiguous at `node_off[node] + 1 : node_off[node] + node_cnt[node]`, keeping their relative
order, and move each row to its child in `nidx`.
"""
function partition_rows!(dst, is, cache, backend)
    n = length(is)
    flag = view(cache.part_flag, 1:n)
    partition_flag_kernel!(backend)(
        flag, is, cache.nidx, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
        ndrange=n,
    )
    # `part_scan[1]` stays 0, making the scan exclusive
    cumsum!(view(cache.part_scan, 2:n+1), flag)
    partition_scatter_kernel!(backend)(
        dst, cache.nidx, cache.node_off, cache.node_cnt, is, flag, cache.part_scan, cache.tree_cond_bin_gpu;
        ndrange=n,
    )
    return nothing
end
"""
    _hist_group(gid, n_tiles, rows_per_group, build_nodes, n_build, node_off, node_cnt, chunk_end)

Node and range `r_lo:r_hi` of `is` read by workgroup `gid`. Groups run over (row chunk, tile) with
the tile varying fastest, and row chunks are numbered across build nodes through the inclusive
prefix `chunk_end`. Groups past the last chunk get an empty range.
"""
@inline function _hist_group(gid, n_tiles, rows_per_group, build_nodes, n_build, node_off, node_cnt, chunk_end)
    q = (gid - 1) ÷ n_tiles
    @inbounds if q >= chunk_end[n_build]
        return 1, 1, 0
    end
    lo, hi = 1, n_build
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        if chunk_end[mid] > q
            hi = mid
        else
            lo = mid + 1
        end
    end
    @inbounds begin
        node = Int(build_nodes[lo])
        c = q - (lo == 1 ? 0 : Int(chunk_end[lo-1]))
        o = Int(node_off[node])
        r_lo = o + c * rows_per_group + 1
        r_hi = min(o + (c + 1) * rows_per_group, o + Int(node_cnt[node]))
    end
    return node, r_lo, r_hi
end

"""
    _hist_tile(gid, n_tiles, n_ftiles, feat_tile, k_tile, n_feats, nk)

Tile of workgroup `gid`: offsets `f0`, `k0` into the features and gradient rows, and its sizes.
"""
@inline function _hist_tile(gid, n_tiles, n_ftiles, feat_tile, k_tile, n_feats, nk)
    t = (gid - 1) % n_tiles
    f0 = (t % n_ftiles) * feat_tile
    k0 = (t ÷ n_ftiles) * k_tile
    return f0, k0, min(feat_tile, n_feats - f0), min(k_tile, nk - k0)
end

"""
    chunk_prefix_kernel!(chunk_end, build_nodes, n_build, node_cnt, rows_per_group)

`chunk_end[b]`: row chunks of `rows_per_group` needed by build nodes `1:b`. Single thread.
"""
@kernel function chunk_prefix_kernel!(chunk_end, @Const(build_nodes), n_build::Int, @Const(node_cnt), rows_per_group::Int)
    i = @index(Global)
    if i == 1
        acc = 0
        @inbounds for b in 1:n_build
            acc += cld(Int(node_cnt[build_nodes[b]]), rows_per_group)
            chunk_end[b] = acc
        end
    end
end

"""
    hist_kernel!(h∇, ∇, x_bin, js, is, build_nodes, n_build, node_off, node_cnt, chunk_end,
                 K, k_tile, feat_tile, rows_per_group, ::Val{LMEM})

Per-node gradient histograms. Each workgroup reads one chunk of one build node's rows of `is`,
accumulates `k_tile` gradient rows of `feat_tile` features in workgroup-local memory, then adds
each non-zero cell to `h∇` with one global atomic. Groups are numbered with the tile varying
fastest, so the launch needs `sum(cld.(node_cnt[build_nodes], rows_per_group)) * n_tiles` groups;
surplus groups do nothing.

- `h∇`: histogram `[2K+1, nbins, n_feats, n_nodes]`, accumulated into.
- `k_tile * nbins * feat_tile <= LMEM`.
"""
@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(js),
    @Const(is),
    @Const(build_nodes),
    n_build::Int,
    @Const(node_off),
    @Const(node_cnt),
    @Const(chunk_end),
    K::Int,
    k_tile::Int,
    feat_tile::Int,
    rows_per_group::Int,
    ::Val{LMEM},
) where {T,LMEM}
    lid = @index(Local, Linear)
    gid = @index(Group, Linear)
    @uniform wg = @groupsize()[1]
    @uniform nk = 2 * K + 1
    @uniform nbins = size(h∇, 2)
    @uniform n_feats = length(js)
    @uniform n_ftiles = cld(n_feats, feat_tile)
    @uniform n_tiles = n_ftiles * cld(nk, k_tile)

    hloc = @localmem T (LMEM,)

    # The CPU backend does not carry plain locals across `@synchronize`, so the group-derived
    # values are recomputed in each phase.
    f0, k0, nf, nkk = _hist_tile(gid, n_tiles, n_ftiles, feat_tile, k_tile, n_feats, nk)
    used = nkk * nbins * nf
    i = lid
    @inbounds while i <= used
        hloc[i] = zero(T)
        i += wg
    end
    @synchronize

    f0, k0, nf, nkk = _hist_tile(gid, n_tiles, n_ftiles, feat_tile, k_tile, n_feats, nk)
    _, r_lo, r_hi = _hist_group(gid, n_tiles, rows_per_group, build_nodes, n_build, node_off, node_cnt, chunk_end)
    r = r_lo + lid - 1
    @inbounds while r <= r_hi
        obs = is[r]
        for fl in 1:nf
            bin = Int(x_bin[obs, js[f0+fl]])
            if bin > 0 && bin <= nbins
                base = nkk * ((bin - 1) + nbins * (fl - 1))
                for kk in 1:nkk
                    Atomix.@atomic hloc[base+kk] += T(∇[k0+kk, obs])
                end
            end
        end
        r += wg
    end
    @synchronize

    f0, k0, nf, nkk = _hist_tile(gid, n_tiles, n_ftiles, feat_tile, k_tile, n_feats, nk)
    used = nkk * nbins * nf
    node, _, _ = _hist_group(gid, n_tiles, rows_per_group, build_nodes, n_build, node_off, node_cnt, chunk_end)
    i = lid
    @inbounds while i <= used
        v = hloc[i]
        if v != zero(T)
            kk = (i - 1) % nkk + 1
            rest = (i - 1) ÷ nkk
            b = rest % nbins + 1
            fl = rest ÷ nbins + 1
            Atomix.@atomic h∇[k0+kk, b, js[f0+fl], node] += v
        end
        i += wg
    end
end

"""
	clear_hist_kernel!(h∇, active_nodes, n_active)

Zero histogram entries in `h∇` for the `n_active` nodes listed in `active_nodes`.
"""
@kernel function clear_hist_kernel!(h∇, @Const(active_nodes), n_active)
    idx = @index(Global, Linear)
    n_elements = size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
    total = n_elements * n_active

    @inbounds if idx <= total
        node_idx = (idx - 1) ÷ n_elements + 1
        element_idx = (idx - 1) % n_elements
        node = active_nodes[node_idx]
        if node > 0
            k = element_idx % size(h∇, 1) + 1
            b = (element_idx ÷ size(h∇, 1)) % size(h∇, 2) + 1
            j = element_idx ÷ (size(h∇, 1) * size(h∇, 2)) + 1
            h∇[k, b, j, node] = zero(eltype(h∇))
        end
    end
end

# Build histograms for `build_nodes`, each from its own range of `is`
function EvoTrees.update_hist!(h∇, ∇, x_bin, js, is, build_nodes, node_off, node_cnt, chunk_end, K, backend)
    n_build = length(build_nodes)
    n_build == 0 && return nothing

    clear_hist_kernel!(backend)(
        h∇, build_nodes, n_build;
        ndrange=n_build * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
    )

    nk = 2 * K + 1
    nbins = size(h∇, 2)
    lmem = EvoTrees.HIST_LMEM
    rows = EvoTrees.HIST_ROWS
    k_tile = min(nk, lmem ÷ nbins)
    feat_tile = min(lmem ÷ (k_tile * nbins), length(js))
    n_tiles = cld(length(js), feat_tile) * cld(nk, k_tile)

    chunk_prefix_kernel!(backend, 1)(chunk_end, build_nodes, n_build, node_cnt, rows; ndrange=1)

    # Upper bound on the chunks, known without reading `chunk_end` back.
    n_groups = (cld(length(is), rows) + n_build) * n_tiles
    hist_kernel!(backend, EvoTrees.HIST_WG)(
        h∇, ∇, x_bin, js, is, build_nodes, n_build, node_off, node_cnt, chunk_end,
        K, k_tile, feat_tile, rows, Val(lmem);
        ndrange=n_groups * EvoTrees.HIST_WG,
    )
    return nothing
end

"""
	separate_nodes_kernel!(build_nodes, build_count, subtract_nodes, subtract_count, active_nodes, nodes_sum)

Split active sibling nodes into:
- **build_nodes**: nodes whose histograms should be built via observation scan (lighter sibling)
- **subtract_nodes**: nodes whose histograms should be computed as `parent - sibling` (heavier sibling)

Node size is the weight sum `nodes_sum[end, node]`, already written by `apply_splits_kernel!`
when the parent was split. Ties are broken by node id.
"""
@kernel function separate_nodes_kernel!(
    build_nodes, build_count,
    subtract_nodes, subtract_count,
    @Const(active_nodes),
    @Const(nodes_sum)
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0
            sibling = node ⊻ 1
            w_node = nodes_sum[end, node]
            w_sibling = nodes_sum[end, sibling]

            if w_node < w_sibling || (w_node == w_sibling && node < sibling)
                pos = Atomix.@atomic build_count[1] += 1
                build_nodes[pos] = node
            else
                pos = Atomix.@atomic subtract_count[1] += 1
                subtract_nodes[pos] = node
            end
        end
    end
end

"""
	subtract_hist_kernel!(h, js, nodes)

Sibling subtraction over `h` reshaped to `(2K+1)*nbins × nfeats × nnodes`.
The 3D ndrange drops the per-element index decode.
"""
@kernel function subtract_hist_kernel!(h, @Const(js), @Const(nodes))
    i, jj, nn = @index(Global, NTuple)
    @inbounds begin
        n = nodes[nn]
        if n > 1
            j = js[jj]
            h[i, j, n] = h[i, j, n>>1] - h[i, j, n⊻1]
        end
    end
end

function EvoTrees.subtract_hist!(h∇::GPUArraysCore.AbstractGPUArray{<:Any,4}, nodes, js)
    backend = get_backend(h∇)
    h = reshape(h∇, :, size(h∇, 3), size(h∇, 4))
    subtract_hist_kernel!(backend)(h, js, nodes; ndrange=(size(h, 1), length(js), length(nodes)))
end

"""
	compute_nodes_sum_kernel!(nodes_sum, h∇, active_nodes, js, K)

Compute per-node gradient totals by summing histograms across bins.
Writes into `nodes_sum[:, node]` for each node in `active_nodes`.
"""
@kernel function compute_nodes_sum_kernel!(nodes_sum, @Const(h∇), @Const(active_nodes), @Const(js), K::Int)
    gidx = @index(Global)
    n_active = length(active_nodes)
    n_k = 2 * K + 1

    @inbounds if gidx <= n_active * n_k
        n_idx = (gidx - 1) ÷ n_k + 1
        k = (gidx - 1) % n_k + 1
        node = active_nodes[n_idx]

        if node > 0
            nbins = size(h∇, 2)
            sum_val = zero(eltype(nodes_sum))
            feat = js[1]
            for b in 1:nbins
                sum_val += h∇[k, b, feat, node]
            end
            nodes_sum[k, node] = sum_val
        end
    end
end

"""
    check_monotone(L, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) -> Bool

Return `true` if the split violates `constraint` and should be skipped.
Always `false` when `constraint == 0`, and for losses that do not support
monotone constraints.
"""
@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.GradientRegression}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.MLE2P}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

@inline check_monotone(::Type{EvoTrees.MLogLoss}, constraint, args...) = false
@inline check_monotone(::Type{EvoTrees.MAE}, constraint, args...) = false
@inline check_monotone(::Type{<:EvoTrees.Quantile}, constraint, args...) = false
@inline check_monotone(::Type{L}, constraint, args...) where {L<:EvoTrees.Cred} = false

"""
    _eval_split_bin(L, h∇, nodes_sum, node, f, b, ...) -> (gain, acc1, acc2, accw)

Advance left-side histogram sums to bin `b` and return net split gain
(`split_gain - gain_p`).

`K == 1` keeps `g`, `h`, `w` in `acc1`, `acc2`, `accw`. `K > 1` writes column
`temp_idx` of `sums_temp`. Ineligible bins return `-Inf` but still update the
accumulators.
"""
Base.@propagate_inbounds function _eval_split_bin(
    ::Type{L},
    h∇,
    nodes_sum,
    node,
    f,
    b,
    is_numeric,
    constraint,
    acc1::T,
    acc2::T,
    accw::T,
    w_p::T,
    gain_p::T,
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    sums_temp,
    temp_idx::Int,
    ε::T,
) where {T,L}
    if K == 1
        acc1, acc2, accw = EvoTrees._accumulate_hist_k1(
            h∇, f, b, node, is_numeric, acc1, acc2, accw,
        )
        w_l, w_r = accw, w_p - accw
        (w_l <= min_weight || w_r <= min_weight) && return (T(-Inf), acc1, acc2, accw)
        check_monotone(
            L, constraint,
            acc1, acc2,
            nodes_sum[1, node] - acc1, nodes_sum[2, node] - acc2,
            w_l, w_r, lambda, L2, ε,
        ) && return (T(-Inf), acc1, acc2, accw)
        ∑ = (nodes_sum[1, node], nodes_sum[2, node], nodes_sum[3, node])
        ∑L = (acc1, acc2, accw)
        gain = EvoTrees.split_gain(L, ∑, ∑L, w_l, w_r, lambda, L2, ε) - gain_p
        return (gain, acc1, acc2, accw)
    else
        EvoTrees._acc_left!(sums_temp, temp_idx, h∇, f, b, node, 2 * K + 1, is_numeric)
        w_l = sums_temp[2*K+1, temp_idx]
        w_r = w_p - w_l
        (w_l <= min_weight || w_r <= min_weight) && return (T(-Inf), acc1, acc2, accw)
        check_monotone(
            L, constraint,
            sums_temp[1, temp_idx], sums_temp[K+1, temp_idx],
            nodes_sum[1, node] - sums_temp[1, temp_idx],
            nodes_sum[K+1, node] - sums_temp[K+1, temp_idx],
            w_l, w_r, lambda, L2, ε,
        ) && return (T(-Inf), acc1, acc2, accw)
        gain = EvoTrees.split_gain(
            L, nodes_sum, node, sums_temp, temp_idx, K, w_l, w_r, lambda, L2, ε,
        ) - gain_p
        return (gain, acc1, acc2, accw)
    end
end

"""
    find_best_split_parallel_kernel!(L, gains, bins, h∇, nodes_sum, active_nodes, js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

One thread per `(active node, feature)`. Write the best bin into `gains[f, n]`
and `bins[f, n]` (`0` if none).
"""
@kernel function find_best_split_parallel_kernel!(
    ::Type{L},
    gains::AbstractMatrix{T},
    bins::AbstractMatrix{Int32},
    @Const(h∇),
    @Const(nodes_sum),
    @Const(active_nodes),
    @Const(js),
    @Const(feattypes),
    @Const(monotone_constraints),
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    n_feats::Int,
    sums_temp::AbstractArray{T,2},
) where {T,L}
    gidx = @index(Global)
    n_active = length(active_nodes)
    ε = T(1e-8)

    @inbounds if gidx <= n_active * n_feats
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node == 0
            gains[f_idx, n_idx] = T(-Inf)
            bins[f_idx, n_idx] = Int32(0)
        else
            f, is_numeric, constraint, w_p, gain_p, b_max = EvoTrees._init_split_scan(
                L, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints,
                lambda, L2, K, ε,
            )
            temp_idx = (n_idx - 1) * n_feats + f_idx
            EvoTrees._clear_split_sums!(sums_temp, temp_idx, K)

            g_best, b_best = T(-Inf), Int32(0)
            acc1, acc2, accw = zero(T), zero(T), zero(T)
            for b in 1:b_max
                g_val, acc1, acc2, accw = _eval_split_bin(
                    L, h∇, nodes_sum, node, f, b, is_numeric, constraint,
                    acc1, acc2, accw, w_p, gain_p,
                    lambda, L2, min_weight, K, sums_temp, temp_idx, ε,
                )
                if g_val > g_best
                    g_best = g_val
                    b_best = Int32(b)
                end
            end

            gains[f_idx, n_idx] = g_best
            bins[f_idx, n_idx] = b_best
        end
    end
end

"""
    accumulate_obliv_gains_kernel!(L, gains_accum, count_accum, h∇, nodes_sum, active_nodes, js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

Sum eligible bin gains across active nodes into `gains_accum[bin, f]` and
increment `count_accum[bin, f]`. A split is valid only when
`count_accum[bin, f] == n_active`.
"""
@kernel function accumulate_obliv_gains_kernel!(
    ::Type{L},
    gains_accum::AbstractMatrix{T},
    count_accum::AbstractMatrix{Int32},
    @Const(h∇),
    @Const(nodes_sum),
    @Const(active_nodes),
    @Const(js),
    @Const(feattypes),
    @Const(monotone_constraints),
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    n_feats::Int,
    sums_temp::AbstractArray{T,2},
) where {T,L}
    gidx = @index(Global)
    n_active = length(active_nodes)
    ε = T(1e-8)

    @inbounds if gidx <= n_active * n_feats
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node != 0
            f, is_numeric, constraint, w_p, gain_p, b_max = EvoTrees._init_split_scan(
                L, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints,
                lambda, L2, K, ε,
            )
            temp_idx = (n_idx - 1) * n_feats + f_idx
            EvoTrees._clear_split_sums!(sums_temp, temp_idx, K)

            acc1, acc2, accw = zero(T), zero(T), zero(T)
            for b in 1:b_max
                g_val, acc1, acc2, accw = _eval_split_bin(
                    L, h∇, nodes_sum, node, f, b, is_numeric, constraint,
                    acc1, acc2, accw, w_p, gain_p,
                    lambda, L2, min_weight, K, sums_temp, temp_idx, ε,
                )
                if g_val > zero(T)
                    Atomix.@atomic gains_accum[b, f_idx] += g_val
                    Atomix.@atomic count_accum[b, f_idx] += Int32(1)
                end
            end
        end
    end
end

"""
	broadcast_obliv_split_kernel!(best_gain, best_bin, best_feat, gain, bin, feat)

Write the shared level split into every active-node `best_*` slot.
"""
@kernel function broadcast_obliv_split_kernel!(best_gain, best_bin, best_feat, gain, bin, feat)
    i = @index(Global)
    @inbounds if i <= length(best_gain)
        best_gain[i] = gain
        best_bin[i] = bin
        best_feat[i] = feat
    end
end

"""
	reduce_best_split_kernel!(best_gain, best_bin, best_feat, gains, bins, js, n_feats)

For each node-column in `gains`, find the feature index with maximum gain and output:
- `best_gain[n_idx]`
- `best_bin[n_idx]`
- `best_feat[n_idx]` (actual feature id from `js`)
"""
@kernel function reduce_best_split_kernel!(
    best_gain,
    best_bin,
    best_feat,
    @Const(gains),
    @Const(bins),
    @Const(js),
    n_feats::Int
)
    n_idx = @index(Global)

    @inbounds if n_idx <= size(gains, 2)
        best_f_idx = 1
        best_g = gains[1, n_idx]

        for f_idx in 2:n_feats
            g = gains[f_idx, n_idx]
            if g > best_g
                best_g = g
                best_f_idx = f_idx
            end
        end

        best_gain[n_idx] = best_g
        best_bin[n_idx] = bins[best_f_idx, n_idx]
        best_feat[n_idx] = js[best_f_idx]
    end
end
