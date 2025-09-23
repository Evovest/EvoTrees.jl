using KernelAbstractions
using Atomix

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
            if bin == 0
                nidx[obs] = zero(T)
            else
                feattype = feattypes[feat]
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + T(Int(!is_left))
            end
        end
    end
end

@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    K::Int,
    chunk_size::Int
) where {T}
    gidx = @index(Global, Linear)
    
    n_feats = length(js)
    n_obs = length(is)
    total_work_items = n_feats * cld(n_obs, chunk_size)
    
    if gidx <= total_work_items
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk = (gidx - 1) ÷ n_feats
        
        feat = js[feat_idx]
        
        start_idx = obs_chunk * chunk_size + 1
        end_idx = min(start_idx + (chunk_size - 1), n_obs)
        
        @inbounds for obs_idx in start_idx:end_idx
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4)
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    for k in 1:(2*K+1)
                        grad = ∇[k, obs]
                        Atomix.@atomic h∇[k, bin, feat, node] += grad
                    end
                end
            end
        end
    end
end

@kernel function find_best_split_from_hist_kernel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(h∇),
    nodes_sum,
    @Const(active_nodes),
    @Const(js),
    @Const(feattypes),
    @Const(monotone_constraints),
    lambda::T,
    min_weight::T,
    K::Int,
    sums_temp::AbstractArray{T,2}
) where {T}
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            gains[n_idx] = T(-Inf)
            bins[n_idx] = Int32(0)
            feats[n_idx] = Int32(0)
        else
            nbins = size(h∇, 2)
            eps = T(1e-8)
            
            # Initialize node sums from first feature - manual sum instead of sum()
            if !isempty(js)
                first_feat = js[1]
                for k in 1:(2*K+1)
                    sum_val = zero(T)
                    for b in 1:nbins
                        sum_val += h∇[k, b, first_feat, node]
                    end
                    nodes_sum[k, node] = sum_val
                end
            end
            
            # Pre-calculate parent gain
            w_p = nodes_sum[2*K+1, node]
            λw = lambda * w_p
            
            gain_p = zero(T)
            if K == 1
                g_p = nodes_sum[1, node]
                h_p = nodes_sum[2, node]
                gain_p = g_p^2 / (h_p + λw + eps)
            else
                for k in 1:K
                    g_k = nodes_sum[k, node]
                    h_k = nodes_sum[K+k, node]
                    gain_p += g_k^2 / (h_k + λw/K + eps)
                end
            end
            
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                # Initialize accumulators directly in sums_temp for K > 1
                if K == 1
                    acc_g, acc_h, acc_w = zero(T), zero(T), zero(T)
                else
                    for kk in 1:(2*K+1)
                        sums_temp[kk, n_idx] = zero(T)
                    end
                end
                
                for b in 1:(nbins - 1)
                    # Update accumulator based on feature type
                    if K == 1
                        if is_numeric
                            acc_g += h∇[1, b, f, node]
                            acc_h += h∇[2, b, f, node]
                            acc_w += h∇[3, b, f, node]
                        else
                            acc_g = h∇[1, b, f, node]
                            acc_h = h∇[2, b, f, node]
                            acc_w = h∇[3, b, f, node]
                        end
                        
                        w_l = acc_w
                        w_r = w_p - w_l
                        
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        g_l, h_l = acc_g, acc_h
                        g_r, h_r = nodes_sum[1, node] - g_l, nodes_sum[2, node] - h_l
                        
                        # Check monotone constraint
                        if constraint != 0
                            pred_l = -g_l/(h_l + lambda*w_l + eps)
                            pred_r = -g_r/(h_r + lambda*w_r + eps)
                            if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                continue
                            end
                        end
                        
                        gain_l = g_l^2 / (h_l + lambda * w_l + eps)
                        gain_r = g_r^2 / (h_r + lambda * w_r + eps)
                        g = gain_l + gain_r - gain_p
                        
                    else  # K > 1
                        if is_numeric
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] += h∇[kk, b, f, node]
                            end
                        else
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] = h∇[kk, b, f, node]
                            end
                        end
                        
                        w_l = sums_temp[2*K+1, n_idx]
                        w_r = w_p - w_l
                        
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        # Check constraint on first class
                        if constraint != 0
                            g_l1 = sums_temp[1, n_idx]
                            h_l1 = sums_temp[K+1, n_idx]
                            g_r1 = nodes_sum[1, node] - g_l1
                            h_r1 = nodes_sum[K+1, node] - h_l1
                            pred_l = -g_l1/(h_l1 + lambda*w_l/K + eps)
                            pred_r = -g_r1/(h_r1 + lambda*w_r/K + eps)
                            if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                continue
                            end
                        end
                        
                        # Calculate total gain for all K classes
                        gain_l = zero(T)
                        gain_r = zero(T)
                        for k in 1:K
                            g_l = sums_temp[k, n_idx]
                            h_l = sums_temp[K+k, n_idx]
                            g_r = nodes_sum[k, node] - g_l
                            h_r = nodes_sum[K+k, node] - h_l
                            
                            gain_l += g_l^2 / (h_l + lambda * w_l / K + eps)
                            gain_r += g_r^2 / (h_r + lambda * w_r / K + eps)
                        end
                        g = gain_l + gain_r - gain_p
                    end
                    
                    # Update best split if better
                    if g > g_best
                        g_best = g
                        b_best = Int32(b)
                        f_best = Int32(f)
                    end
                end
            end
            
            gains[n_idx] = g_best
            bins[n_idx] = b_best
            feats[n_idx] = f_best
        end
    end
end

@kernel function separate_nodes_kernel!(
    build_nodes, build_count,
    subtract_nodes, subtract_count,
    @Const(active_nodes)
)
    idx = @index(Global)
    @inbounds node = active_nodes[idx]
    
    if node > 0
        if idx % 2 == 1
            pos = Atomix.@atomic build_count[1] += 1
            build_nodes[pos] = node
        else
            pos = Atomix.@atomic subtract_count[1] += 1
            subtract_nodes[pos] = node
        end
    end
end

@kernel function subtract_hist_kernel!(h∇, @Const(subtract_nodes))
    gidx = @index(Global)

    n_k = size(h∇, 1)
    n_b = size(h∇, 2)
    n_j = size(h∇, 3)
    n_elements_per_node = n_k * n_b * n_j

    node_idx = (gidx - 1) ÷ n_elements_per_node + 1
    
    if node_idx <= length(subtract_nodes)
        remainder = (gidx - 1) % n_elements_per_node
        j = remainder ÷ (n_k * n_b) + 1
        
        remainder = remainder % (n_k * n_b)
        b = remainder ÷ n_k + 1
        
        k = remainder % n_k + 1
        
        @inbounds node = subtract_nodes[node_idx]
        
        if node > 0
            parent = node >> 1
            sibling = node ⊻ 1
            
            @inbounds h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    feattypes, monotone_constraints, K, sums_temp=nothing
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if sums_temp === nothing && K > 1
        sums_temp = similar(nodes_sum_gpu, 2*K+1, max(n_active, 1))
    elseif K == 1
        sums_temp = similar(nodes_sum_gpu, 1, 1)
    end
    
    h∇ .= 0
    
    n_feats = length(js)
    chunk_size = 64
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = n_feats * n_obs_chunks
    
    hist_kernel_f! = hist_kernel!(backend)
    workgroup_size = min(256, max(64, num_threads))
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size; ndrange = num_threads, workgroupsize = workgroup_size)
    KernelAbstractions.synchronize(backend)
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js, feattypes, monotone_constraints,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight), K, sums_temp;
                ndrange = max(n_active, 1), workgroupsize = min(256, max(64, n_active)))
    KernelAbstractions.synchronize(backend)
end
