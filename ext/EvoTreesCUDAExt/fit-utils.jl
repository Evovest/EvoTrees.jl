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
function update_hist_gpu!(h∇, ∇, x_bin, nidx, js, is, active_nodes, K, target_mask, backend)
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

    chunk_size = 16
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
	subtract_hist_kernel!(h∇, subtract_nodes)

Compute histograms for nodes in `subtract_nodes` via subtraction:
`h∇[:,:,:,child] = h∇[:,:,:,parent] - h∇[:,:,:,sibling]`.
"""
@kernel function subtract_hist_kernel!(h∇, @Const(subtract_nodes))
    gidx = @index(Global)
    n_k, n_b, n_j = size(h∇, 1), size(h∇, 2), size(h∇, 3)
    n_elements_per_node = n_k * n_b * n_j
    node_idx = (gidx - 1) ÷ n_elements_per_node + 1

    @inbounds if node_idx <= length(subtract_nodes)
        remainder = (gidx - 1) % n_elements_per_node
        j = remainder ÷ (n_k * n_b) + 1
        remainder = remainder % (n_k * n_b)
        b = remainder ÷ n_k + 1
        k = remainder % n_k + 1

        node = subtract_nodes[node_idx]
        if node > 0
            parent = node >> 1
            sibling = node ⊻ 1
            h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

"""
	reduce_root_sums_kernel!(nodes_sum, ∇, is)

Accumulate (atomic) gradient sums over observations `is` into the root node (node id = 1).
"""
@kernel function reduce_root_sums_kernel!(nodes_sum, @Const(∇), @Const(is))
    idx = @index(Global)
    @inbounds if idx <= length(is)
        obs = is[idx]
        n_k = size(∇, 1)
        for k in 1:n_k
            Atomix.@atomic nodes_sum[k, 1] += ∇[k, obs]
        end
    end
end

"""
	compute_nodes_sum_kernel!(nodes_sum, h∇, active_nodes, K)

Compute per-node gradient totals by summing histograms across bins.
Writes into `nodes_sum[:, node]` for each node in `active_nodes`.
"""
@kernel function compute_nodes_sum_kernel!(nodes_sum, @Const(h∇), @Const(active_nodes), K::Int)
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
            for b in 1:nbins
                sum_val += h∇[k, b, 1, node]
            end
            nodes_sum[k, node] = sum_val
        end
    end
end

# Split statistics for gain computation
struct SplitStats{T}
    g_l::T
    h_l::T
    w_l::T
    g_r::T
    h_r::T
    w_r::T
    g_p::T
    h_p::T
    w_p::T
end

# Parent gain: GradientRegression
@inline function parent_gain(::Type{L}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T,L<:EvoTrees.GradientRegression}
    if K == 1
        g, h = nodes_sum[1, node], nodes_sum[2, node]
        return g^2 / max(h + λw + L2, ε) / 2
    else
        gain = zero(T)
        for k in 1:K
            g, h = nodes_sum[k, node], nodes_sum[K+k, node]
            gain += g^2 / max(h + λw + L2, ε)
        end
        return gain / 2
    end
end

# Parent gain: MLE2P
@inline function parent_gain(::Type{L}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T,L<:EvoTrees.MLE2P}
    g1, g2 = nodes_sum[1, node], nodes_sum[2, node]
    h1, h2 = nodes_sum[3, node], nodes_sum[4, node]
    d1 = max(h1 + λw + L2, ε)
    d2 = max(h2 + λw + L2, ε)
    return (g1^2 / d1 + g2^2 / d2) / 2
end

# Parent gain: MLogLoss
@inline function parent_gain(::Type{EvoTrees.MLogLoss}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T}
    gain = zero(T)
    for k in 1:K
        g, h = nodes_sum[k, node], nodes_sum[K+k, node]
        gain += g^2 / max(h + λw + L2, ε)
    end
    return gain / 2
end

# Parent gain: MAE
@inline function parent_gain(::Type{EvoTrees.MAE}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T}
    return zero(T)
end

# Parent gain: Quantile
@inline function parent_gain(::Type{EvoTrees.Quantile}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T}
    return zero(T)
end

# Parent gain: Cred
@inline function parent_gain(::Type{L}, nodes_sum, node, K, λw, L2, w_p, ε::T) where {T,L<:EvoTrees.Cred}
    μ = nodes_sum[1, node] / w_p
    VHM = μ^2
    EVPV = max(nodes_sum[2, node] / w_p - VHM, ε)
    Z = VHM / (VHM + EVPV)
    return Z * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
end

# Split gain: GradientRegression
@inline function split_gain(::Type{L}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T,L<:EvoTrees.GradientRegression}
    d_l = max(s.h_l + lambda * s.w_l + L2, ε)
    d_r = max(s.h_r + lambda * s.w_r + L2, ε)
    return (s.g_l^2 / d_l + s.g_r^2 / d_r) / 2 - gain_p
end

# Split gain: MLE2P
@inline function split_gain(::Type{L}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T,L<:EvoTrees.MLE2P}
    d_l = max(s.h_l + lambda * s.w_l + L2, ε)
    d_r = max(s.h_r + lambda * s.w_r + L2, ε)
    return (s.g_l^2 / d_l + s.g_r^2 / d_r) / 2 - gain_p
end

# Split gain: MLogLoss
@inline function split_gain(::Type{EvoTrees.MLogLoss}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T}
    d_l = max(s.h_l + lambda * s.w_l + L2, ε)
    d_r = max(s.h_r + lambda * s.w_r + L2, ε)
    return (s.g_l^2 / d_l + s.g_r^2 / d_r) / 2 - gain_p
end

# Split gain: MAE
@inline function split_gain(::Type{EvoTrees.MAE}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T}
    μp = s.g_p / s.w_p
    μl = s.g_l / s.w_l
    μr = s.g_r / s.w_r
    d_l = max(1 + lambda + L2 / s.w_l, ε)
    d_r = max(1 + lambda + L2 / s.w_r, ε)
    return abs(μl - μp) * s.w_l / d_l + abs(μr - μp) * s.w_r / d_r
end

# Split gain: Quantile
@inline function split_gain(::Type{EvoTrees.Quantile}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T}
    μp = s.g_p / s.w_p
    μl = s.g_l / s.w_l
    μr = s.g_r / s.w_r
    d_l = max(1 + lambda + L2 / s.w_l, ε)
    d_r = max(1 + lambda + L2 / s.w_r, ε)
    return abs(μl - μp) * s.w_l / d_l + abs(μr - μp) * s.w_r / d_r
end

# Split gain: Cred
@inline function split_gain(::Type{L}, s::SplitStats{T}, gain_p, lambda, L2, ε) where {T,L<:EvoTrees.Cred}
    μl = s.g_l / s.w_l
    VHM_l = μl^2
    EVPV_l = max(s.h_l / s.w_l - VHM_l, ε)
    Z_l = VHM_l / (VHM_l + EVPV_l)
    gain_l = Z_l * abs(s.g_l) / (1 + L2 / s.w_l)

    μr = s.g_r / s.w_r
    VHM_r = μr^2
    EVPV_r = max(s.h_r / s.w_r - VHM_r, ε)
    Z_r = VHM_r / (VHM_r + EVPV_r)
    gain_r = Z_r * abs(s.g_r) / (1 + L2 / s.w_r)

    return gain_l + gain_r - gain_p
end

# Split gain for K>1: GradientRegression/MLE2P/MLogLoss
@inline function split_gain_multi(
    ::Type{L}, sums_temp, nodes_sum, node, temp_idx,
    K, w_l, w_r, gain_p, lambda, L2, ε::T
) where {T,L<:Union{EvoTrees.GradientRegression,EvoTrees.MLE2P,EvoTrees.MLogLoss}}
    g_val = zero(T)
    for k in 1:K
        g_l = sums_temp[k, temp_idx]
        h_l = sums_temp[K+k, temp_idx]
        g_r = nodes_sum[k, node] - g_l
        h_r = nodes_sum[K+k, node] - h_l
        d_l = max(h_l + lambda * w_l + L2, ε)
        d_r = max(h_r + lambda * w_r + L2, ε)
        g_val += (g_l^2 / d_l + g_r^2 / d_r) / 2
    end
    return g_val - gain_p
end

# Monotone constraint check: GradientRegression
@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.GradientRegression}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

# Monotone constraint check: MLE2P
@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.MLE2P}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

# Monotone constraint check: MLogLoss (no constraints)
@inline check_monotone(::Type{EvoTrees.MLogLoss}, constraint, args...) = false

# Monotone constraint check: MAE (no constraints)
@inline check_monotone(::Type{EvoTrees.MAE}, constraint, args...) = false

# Monotone constraint check: Quantile (no constraints)
@inline check_monotone(::Type{EvoTrees.Quantile}, constraint, args...) = false

# Monotone constraint check: Cred (no constraints)
@inline check_monotone(::Type{L}, constraint, args...) where {L<:EvoTrees.Cred} = false

# Accumulate histogram for K=1
@inline function accumulate_hist_k1(h∇, f, b, node, is_numeric, acc1, acc2, accw)
    if is_numeric
        return (acc1 + h∇[1, b, f, node], acc2 + h∇[2, b, f, node], accw + h∇[3, b, f, node])
    else
        return (h∇[1, b, f, node], h∇[2, b, f, node], h∇[3, b, f, node])
    end
end

# Accumulate histogram for K>1
@inline function accumulate_hist_kn!(sums_temp, h∇, f, b, node, K, is_numeric, temp_idx)
    @inbounds for kk in 1:(2*K+1)
        if is_numeric
            sums_temp[kk, temp_idx] += h∇[kk, b, f, node]
        else
            sums_temp[kk, temp_idx] = h∇[kk, b, f, node]
        end
    end
end

"""
	find_best_split_parallel_kernel!(L, gains, bins, h∇, nodes_sum, active_nodes, js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

Evaluate all candidate split bins for each (active node, feature) pair and store:
- `gains[f_idx, n_idx]`: best gain
- `bins[f_idx, n_idx]`: best bin threshold/category id (0 if none)

Notes:
- Numeric features scan bins `1:nbins-1` (cumulative left).
- Categorical features scan bins `1:nbins` (one-vs-rest per bin).
- Monotone constraints apply only to `GradientRegression` and `MLE2P` losses.
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
        # Decode thread index into (node, feature) pair
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node == 0
            # Invalid node, mark as no valid split
            gains[f_idx, n_idx] = T(-Inf)
            bins[f_idx, n_idx] = Int32(0)
        else
            # Setup feature info and parent weight
            f = js[f_idx]
            nbins = size(h∇, 2)
            is_numeric = feattypes[f]
            constraint = monotone_constraints[f]
            w_p = nodes_sum[2*K+1, node]
            λw = lambda * w_p

            # Compute baseline gain before splitting
            gain_p = parent_gain(L, nodes_sum, node, K, λw, L2, w_p, ε)

            # Track best split found for this (node, feature)
            g_best = T(-Inf)
            b_best = Int32(0)

            # Unique temp storage column for this thread
            temp_idx = (n_idx - 1) * n_feats + f_idx

            # Initialize left-side accumulators (cumulative sums)
            acc1, acc2, accw = zero(T), zero(T), zero(T)
            if K > 1
                for kk in 1:(2*K+1)
                    sums_temp[kk, temp_idx] = zero(T)
                end
            end

            # Scan bins: numeric excludes last bin, categorical includes all
            b_max = is_numeric ? (nbins - 1) : nbins
            for b in 1:b_max
                if K == 1
                    # Accumulate histogram into left-side sums
                    acc1, acc2, accw = accumulate_hist_k1(h∇, f, b, node, is_numeric, acc1, acc2, accw)
                    w_l, w_r = accw, w_p - accw

                    # Skip if either child has insufficient weight
                    (w_l < min_weight || w_r < min_weight) && continue

                    # Compute right-side stats by subtraction
                    g_l, h_l = acc1, acc2
                    g_r = nodes_sum[1, node] - g_l
                    h_r = nodes_sum[2, node] - h_l
                    g_p = nodes_sum[1, node]
                    h_p = nodes_sum[2, node]

                    # Skip if split violates monotone constraint
                    check_monotone(L, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) && continue

                    # Compute split gain using loss-specific formula
                    stats = SplitStats(g_l, h_l, w_l, g_r, h_r, w_r, g_p, h_p, w_p)
                    g_val = split_gain(L, stats, gain_p, lambda, L2, ε)
                else
                    # K > 1: accumulate all gradient components
                    accumulate_hist_kn!(sums_temp, h∇, f, b, node, K, is_numeric, temp_idx)
                    w_l = sums_temp[2*K+1, temp_idx]
                    w_r = w_p - w_l

                    # Skip if either child has insufficient weight
                    (w_l < min_weight || w_r < min_weight) && continue

                    # Check monotone constraint using first output dimension
                    g_l1 = sums_temp[1, temp_idx]
                    h_l1 = sums_temp[K+1, temp_idx]
                    g_r1 = nodes_sum[1, node] - g_l1
                    h_r1 = nodes_sum[K+1, node] - h_l1
                    check_monotone(L, constraint, g_l1, h_l1, g_r1, h_r1, w_l, w_r, lambda, L2, ε) && continue

                    # Compute multi-output split gain
                    g_val = split_gain_multi(L, sums_temp, nodes_sum, node, temp_idx, K, w_l, w_r, gain_p, lambda, L2, ε)
                end

                # Update best if this split is better
                if g_val > g_best
                    g_best = g_val
                    b_best = Int32(b)
                end
            end

            # Store best split for this (node, feature) pair
            gains[f_idx, n_idx] = g_best
            bins[f_idx, n_idx] = b_best
        end
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

