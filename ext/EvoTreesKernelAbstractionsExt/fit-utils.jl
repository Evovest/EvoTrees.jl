using KernelAbstractions
using Atomix

"""
	update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)

Update observation-to-node assignments by traversing splits (left child = node*2, right child = node*2+1).
"""
@kernel function update_nodes_idx_kernel!(
    nidx::AbstractVector{T},
    @Const(is),
    @Const(x_bin),
    @Const(cond_feats),
    @Const(cond_bins),
    @Const(feattypes),
) where {T<:Unsigned}
    gidx = @index(Global)
    @inbounds if gidx <= length(is)
        obs = is[gidx]
        node = nidx[obs]
        if node > 0
            feat = cond_feats[node]
            bin = cond_bins[node]
            if bin != 0
                feattype = feattypes[feat]
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + T(Int(!is_left))
            end
        end
    end
end

"""
	count_nodes_kernel!(node_counts, nidx, is)

Count the number of observations assigned to each node (raw counts), using atomic increments.
"""
@kernel function count_nodes_kernel!(node_counts, @Const(nidx), @Const(is))
    idx = @index(Global)
    @inbounds if idx <= length(is)
        obs = is[idx]
        node = nidx[obs]
        if node > 0 && node <= length(node_counts)
            Atomix.@atomic node_counts[node] += 1
        end
    end
end

"""
	hist_kernel!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size, target_mask)

Build per-node gradient histograms using atomic updates.

- `h∇` layout: [2K+1, nbins, n_feats, n_nodes]
- Each thread processes one (feature, observation-chunk) pair to reduce contention.
"""
@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    K::Int,
    chunk_size::Int,
    @Const(target_mask)
) where {T}
    gidx = @index(Global, Linear)
    n_feats = length(js)
    n_obs = length(is)
    total_chunks = cld(n_obs, chunk_size)
    total_threads = n_feats * total_chunks

    @inbounds if gidx <= total_threads
        feat_idx = (gidx - 1) % n_feats + 1
        chunk_idx = (gidx - 1) ÷ n_feats
        feat = js[feat_idx]

        start_obs = chunk_idx * chunk_size + 1
        end_obs = min(start_obs + chunk_size - 1, n_obs)

        for obs_idx in start_obs:end_obs
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4) && target_mask[node] != 0
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    for k in 1:(2*K+1)
                        Atomix.@atomic h∇[k, bin, feat, node] += ∇[k, obs]
                    end
                end
            end
        end
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

"""
	clear_mask_kernel!(mask)

Set all entries of `mask` to 0.
"""
@kernel function clear_mask_kernel!(mask)
    idx = @index(Global)
    @inbounds if idx <= length(mask)
        mask[idx] = 0
    end
end

"""
	mark_active_nodes_kernel!(mask, active_nodes)

Mark each node id in `active_nodes` as active by setting `mask[node] = 1`.
"""
@kernel function mark_active_nodes_kernel!(mask, @Const(active_nodes))
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0 && node <= length(mask)
            mask[node] = 1
        end
    end
end

# Build histograms for active nodes
function EvoTrees.update_hist!(h∇, ∇, x_bin, nidx, js, is, active_nodes, K, target_mask, backend)
    n_active = length(active_nodes)

    clear_mask_kernel!(backend)(target_mask; ndrange=length(target_mask))
    KernelAbstractions.synchronize(backend)

    mark_active_nodes_kernel!(backend)(target_mask, active_nodes; ndrange=n_active)
    KernelAbstractions.synchronize(backend)

    if n_active > 0
        clear_hist_kernel!(backend)(
            h∇, active_nodes, n_active;
            ndrange=n_active * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
        )
        KernelAbstractions.synchronize(backend)
    end

    chunk_size = EvoTrees.HIST_OBS_CHUNK
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = length(js) * n_obs_chunks

    hist_kernel!(backend)(
        h∇, ∇, x_bin, nidx, js, is, K, chunk_size, target_mask;
        ndrange=num_threads,
    )
    KernelAbstractions.synchronize(backend)
end

"""
	separate_nodes_kernel!(build_nodes, build_count, subtract_nodes, subtract_count, active_nodes, node_counts)

Split active sibling nodes into:
- **build_nodes**: nodes whose histograms should be built via observation scan (smaller sibling)
- **subtract_nodes**: nodes whose histograms should be computed as `parent - sibling` (larger sibling)

Ties are broken by node id.
"""
@kernel function separate_nodes_kernel!(
    build_nodes, build_count,
    subtract_nodes, subtract_count,
    @Const(active_nodes),
    @Const(node_counts)
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0
            sibling = node ⊻ 1
            w_node = node_counts[node]
            w_sibling = node_counts[sibling]

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
    KernelAbstractions.synchronize(backend)
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
        (w_l < min_weight || w_r < min_weight) && return (T(-Inf), acc1, acc2, accw)
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
        (w_l < min_weight || w_r < min_weight) && return (T(-Inf), acc1, acc2, accw)
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
