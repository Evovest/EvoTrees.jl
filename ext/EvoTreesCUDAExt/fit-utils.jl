"""
    update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
"""
@kernel function update_nodes_idx_kernel_ka!(
    nidx::AbstractVector{T},
    @Const(is),
    @Const(x_bin),
    @Const(cond_feats),
    @Const(cond_bins),
    @Const(feattypes)
) where {T}
    gidx = @index(Global)

    @inbounds if gidx <= length(is)
        obs  = is[gidx]
        node = nidx[obs]

        if node != 0
            feat = cond_feats[node]
            bin  = cond_bins[node]

            if bin == 0
                nidx[obs] = zero(T)
            else
                feattype = feattypes[feat]
                is_left  = feattype ? x_bin[obs, feat] <= bin : x_bin[obs, feat] == bin
                nidx[obs] = T((node << 1) + Int32(!is_left))
            end
        else
            nidx[obs] = zero(T)
        end
    end
end

"""
    update_nodes_idx_gpu!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)

# Arguments

- nidx: vector of Int of length == nobs indicating the active node of each observation
- is: vector or observations id
- x_bin: binarized feature matrix
- cond_feats: vector of Int indicating the feature on which to perform the condition. Length == #nodes
- cond_bins: vector of Float indicating the value for splitting a feature. Length == #nodes
- feattypes: vector of Bool indicating whether the feature is ordinal

"""
function update_nodes_idx_gpu!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    backend = KernelAbstractions.get_backend(nidx)

    kernel! = update_nodes_idx_kernel_ka!(backend)

    kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes; ndrange = length(is))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ===== MINIMAL OPTIMIZED FUNCTIONS =====

@kernel function hist_kernel_ka!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js)
) where {T}
    i, j = @index(Global, NTuple)
    
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0
            jdx = js[j]
            bin = x_bin[i, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

function apply_hist_subtraction!(h∇, dnodes)
    for n in dnodes
        if n % 2 == 0  # left child
            parent = n >> 1
            right = n + 1
            if right in dnodes
                @inbounds h∇[:, :, :, right] = h∇[:, :, :, parent] .- h∇[:, :, :, n]
            end
        end
    end
end

function update_hist_gpu_optimized!(h∇, ∇, x_bin, is, js, nidx, depth, anodes)
    backend = KernelAbstractions.get_backend(h∇)
    dnodes = 2^(depth-1):2^depth-1
    
    # Single broadcast kernel instead of one memset per node
    @inbounds h∇[:, :, :, dnodes] .= 0
    
    kernel! = hist_kernel_ka!(backend)
    kernel!(h∇, ∇, x_bin, nidx, js; ndrange=(size(x_bin, 1), length(js)))
    KernelAbstractions.synchronize(backend)
    
    if depth > 1
        apply_hist_subtraction!(h∇, dnodes)
    end
    return nothing
end

# === New helper kernel: build the list of active node ids directly on device ===
@kernel function fill_active_nodes_kernel!(active_nodes::AbstractVector{Int32}, offset::Int32)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        # Node ids for the current depth start at `1 + offset`
        active_nodes[idx] = Int32(idx + offset)
    end
end

function fill_active_nodes_gpu!(active_nodes, offset::Int32, backend)
    kernel! = fill_active_nodes_kernel!(backend)
    kernel!(active_nodes, offset; ndrange = length(active_nodes))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# === New kernel: compute best split directly from histograms (one thread per node) ===
@kernel function best_split_hist_kernel!(
    best_gain::AbstractVector{T},
    best_bin::AbstractVector{Int32},
    best_feat::AbstractVector{Int32},
    @Const(h∇),
    @Const(active_nodes),
    lambda::T,
    min_weight::T,
) where {T}
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        nbins  = size(h∇, 2)
        nfeats = size(h∇, 3)

        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)

        # iterate feature → bins
        for f in 1:nfeats
            # parent stats (aggregate over all bins once per feature)
            p_g1 = zero(T); p_g2 = zero(T); p_w = zero(T)
            @inbounds for b in 1:nbins
                p_g1 += h∇[1, b, f, node]
                p_g2 += h∇[2, b, f, node]
                p_w  += h∇[3, b, f, node]
            end
            # cumulative left stats
            l_g1 = zero(T); l_g2 = zero(T); l_w = zero(T)
            @inbounds for b in 1:(nbins - 1)
                l_g1 += h∇[1, b, f, node]
                l_g2 += h∇[2, b, f, node]
                l_w  += h∇[3, b, f, node]
                r_w = p_w - l_w
                if l_w >= min_weight && r_w >= min_weight
                    r_g1 = p_g1 - l_g1
                    r_g2 = p_g2 - l_g2
                    gain_l = l_g1^2 / (l_g2 + lambda)
                    gain_r = r_g1^2 / (r_g2 + lambda)
                    gain_p = p_g1^2 / (p_g2 + lambda)
                    g = (gain_l + gain_r - gain_p) / T(2)
                    if g > g_best
                        g_best = g
                        b_best = Int32(b)
                        f_best = Int32(f)
                    end
                end
            end
        end
        best_gain[n_idx] = g_best
        best_bin[n_idx]  = b_best
        best_feat[n_idx] = f_best
    end
end

# === New kernel: build left (prefix) and right (suffix) cumulative histograms in a single pass ===
@kernel function prefix_suffix_kernel!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    @Const(active_nodes)
) where {T}
    n_idx, feat = @index(Global, NTuple)
    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node   = active_nodes[n_idx]
        nbins  = size(h∇, 2)
        nstats = size(h∇, 1)   # 3 for most losses, 5 for MLE

        if nstats == 3
            # ---- unrolled 3-stat path (g1, g2, w) ----
            tot1 = zero(T); tot2 = zero(T); tot3 = zero(T)
            @inbounds for b in 1:nbins
                tot1 += h∇[1, b, feat, node]
                tot2 += h∇[2, b, feat, node]
                tot3 += h∇[3, b, feat, node]
            end
            run1 = zero(T); run2 = zero(T); run3 = zero(T)
            @inbounds for b in 1:nbins
                run1 += h∇[1, b, feat, node]; hL[1, b, feat, node] = run1; hR[1, b, feat, node] = tot1 - run1
                run2 += h∇[2, b, feat, node]; hL[2, b, feat, node] = run2; hR[2, b, feat, node] = tot2 - run2
                run3 += h∇[3, b, feat, node]; hL[3, b, feat, node] = run3; hR[3, b, feat, node] = tot3 - run3
            end
        else
            # ---- unrolled 5-stat path (MLE losses) ----
            tot1 = zero(T); tot2 = zero(T); tot3 = zero(T); tot4 = zero(T); tot5 = zero(T)
            @inbounds for b in 1:nbins
                tot1 += h∇[1, b, feat, node]
                tot2 += h∇[2, b, feat, node]
                tot3 += h∇[3, b, feat, node]
                tot4 += h∇[4, b, feat, node]
                tot5 += h∇[5, b, feat, node]
            end
            run1 = zero(T); run2 = zero(T); run3 = zero(T); run4 = zero(T); run5 = zero(T)
            @inbounds for b in 1:nbins
                run1 += h∇[1, b, feat, node]; hL[1, b, feat, node] = run1; hR[1, b, feat, node] = tot1 - run1
                run2 += h∇[2, b, feat, node]; hL[2, b, feat, node] = run2; hR[2, b, feat, node] = tot2 - run2
                run3 += h∇[3, b, feat, node]; hL[3, b, feat, node] = run3; hR[3, b, feat, node] = tot3 - run3
                run4 += h∇[4, b, feat, node]; hL[4, b, feat, node] = run4; hR[4, b, feat, node] = tot4 - run4
                run5 += h∇[5, b, feat, node]; hL[5, b, feat, node] = run5; hR[5, b, feat, node] = tot5 - run5
            end
        end
    end
end

function compute_lr_gpu!(hL, hR, h∇, active_nodes, backend)
    kernel! = prefix_suffix_kernel!(backend)
    kernel!(hL, hR, h∇, active_nodes; ndrange=(length(active_nodes), size(h∇, 3)))
    KernelAbstractions.synchronize(backend)
    return nothing
end

