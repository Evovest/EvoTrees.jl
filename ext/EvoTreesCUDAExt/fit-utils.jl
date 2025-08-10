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

@kernel function compute_gains_kernel!(
    gains::AbstractArray{T,3},
    @Const(h∇),
    @Const(active_nodes),
    lambda::T,
    min_weight::T
) where {T}
    n_idx, feat = @index(Global, NTuple)
    
    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        n = active_nodes[n_idx]
        nbins = size(h∇, 2)
        # parent stats
        p_g1 = zero(T); p_g2 = zero(T); p_w = zero(T)
        @inbounds for bin in 1:nbins
            p_g1 += h∇[1, bin, feat, n]
            p_g2 += h∇[2, bin, feat, n]
            p_w  += h∇[3, bin, feat, n]
        end
        # cumulative left stats
        l_g1 = zero(T); l_g2 = zero(T); l_w = zero(T)
        @inbounds for bin in 1:(nbins-1)
            l_g1 += h∇[1, bin, feat, n]
            l_g2 += h∇[2, bin, feat, n]
            l_w  += h∇[3, bin, feat, n]
            r_w  = p_w - l_w
            if l_w >= min_weight && r_w >= min_weight
                r_g1 = p_g1 - l_g1
                r_g2 = p_g2 - l_g2
                gain_l = l_g1^2 / (l_g2 + lambda)
                gain_r = r_g1^2 / (r_g2 + lambda)
                gain_p = p_g1^2 / (p_g2 + lambda)
                gains[bin, feat, n] = (gain_l + gain_r - gain_p) / T(2)
            else
                gains[bin, feat, n] = T(0)
            end
        end
        # last bin cannot split
        gains[nbins, feat, n] = T(0)
    end
end

@kernel function best_split_kernel!(
    best_gain::AbstractArray{T,1},
    best_bin::AbstractVector{Int32},
    best_feat::AbstractVector{Int32},
    @Const(gains),
    @Const(active_nodes)
) where {T}
    n_idx = @index(Global)

    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        nbins = size(gains, 1)
        nfeats = size(gains, 2)

        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)

        @inbounds for f in 1:nfeats
            @inbounds for b in 1:nbins
                g = gains[b, f, node]
                if g > g_best
                    g_best = g
                    b_best = Int32(b)
                    f_best = Int32(f)
                end
            end
        end

        best_gain[n_idx] = g_best
        best_bin[n_idx]  = b_best
        best_feat[n_idx] = f_best
    end
end

