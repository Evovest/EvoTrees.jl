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

@kernel function fill_mask_kernel!(mask::AbstractVector{UInt8}, @Const(nodes))
    i = @index(Global)
    @inbounds if i <= length(nodes)
        node = nodes[i]
        if node > 0 && node <= length(mask)
            mask[node] = UInt8(1)
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
    K::Int
) where {T}
    gidx = @index(Global, Linear)
    
    n_feats = length(js)
    n_obs = length(is)
    total_work_items = n_feats * cld(n_obs, 12)
    
    if gidx <= total_work_items
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk = (gidx - 1) ÷ n_feats
        
        feat = js[feat_idx]
        
        start_idx = obs_chunk * 12 + 1
        end_idx = min(start_idx + 11, n_obs)
        
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
    K::Int
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
            
            for k in 1:(2*K+1)
                sum_val = zero(T)
                for j_idx in 1:length(js)
                    f = js[j_idx]
                    for b in 1:nbins
                        sum_val += h∇[k, b, j_idx, node]
                    end
                end
                nodes_sum[k, node] = sum_val
            end
            
            gain_p = zero(T)
            w_p = nodes_sum[2*K+1, node]
            if K == 1
                g_p = nodes_sum[1, node]
                h_p = nodes_sum[2, node]
                gain_p = g_p^2 / (h_p + lambda * w_p + T(1e-8))
            else
                for k in 1:K
                    g_p = nodes_sum[k, node]
                    h_p = nodes_sum[K+k, node]
                    gain_p += g_p^2 / (h_p + lambda * w_p / K + T(1e-8))
                end
            end
            
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)
            
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                if is_numeric  
                    s1, s2, s3 = zero(T), zero(T), zero(T)
                    for b in 1:(nbins - 1)
                        s1 += h∇[1, b, j_idx, node]
                        s2 += h∇[2, b, j_idx, node]  
                        s3 += h∇[2*K+1, b, j_idx, node]
                        
                        if s3 >= min_weight && (w_p - s3) >= min_weight
                            l_g1, l_g2 = s1, s2
                            r_g1, r_g2 = nodes_sum[1, node] - l_g1, nodes_sum[2, node] - l_g2
                            
                            if constraint != 0
                                predL = -l_g1 / (l_g2 + lambda * s3 + T(1e-8))
                                predR = -r_g1 / (r_g2 + lambda * (w_p - s3) + T(1e-8))
                                
                                if !((constraint == 0) || (constraint == -1 && predL > predR) || (constraint == 1 && predL < predR))
                                    continue
                                end
                            end
                            
                            gain_l = l_g1^2 / (s3 * lambda + l_g2 + T(1e-8))
                            gain_r = r_g1^2 / ((w_p - s3) * lambda + r_g2 + T(1e-8))
                            
                            g = gain_l + gain_r - gain_p
                            if g > g_best
                                g_best = g
                                b_best = Int32(b)
                                f_best = Int32(f)
                            end
                        end
                    end
                else  
                    for b in 1:(nbins - 1)
                        l_g1 = h∇[1, b, j_idx, node]
                        l_g2 = h∇[2, b, j_idx, node]
                        l_w = h∇[2*K+1, b, j_idx, node]
                        
                        r_g1 = nodes_sum[1, node] - l_g1
                        r_g2 = nodes_sum[2, node] - l_g2
                        r_w = w_p - l_w
                        
                        if l_w >= min_weight && r_w >= min_weight
                            if constraint != 0
                                predL = -l_g1 / (l_g2 + lambda * l_w + T(1e-8))
                                predR = -r_g1 / (r_g2 + lambda * r_w + T(1e-8))
                                
                                if !((constraint == 0) || (constraint == -1 && predL > predR) || (constraint == 1 && predL < predR))
                                    continue
                                end
                            end
                            
                            gain_l = l_g1^2 / (l_w * lambda + l_g2 + T(1e-8))
                            gain_r = r_g1^2 / (r_w * lambda + r_g2 + T(1e-8))
                            
                            g = gain_l + gain_r - gain_p
                            if g > g_best
                                g_best = g
                                b_best = Int32(b)
                                f_best = Int32(f)
                            end
                        end
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
    left_nodes_buf, right_nodes_buf, target_mask_buf, feattypes, monotone_constraints, K
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    h∇ .= 0
    
    n_feats = length(js)
    n_obs_chunks = cld(length(is), 12)
    num_threads = n_feats * n_obs_chunks
    
    hist_kernel_f! = hist_kernel!(backend)
    workgroup_size = min(256, max(32, num_threads))
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is, K; ndrange = num_threads, workgroupsize = workgroup_size)
    if n_active > 1
        KernelAbstractions.synchronize(backend)
    end
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js, feattypes, monotone_constraints,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight), K;
                ndrange = max(n_active, 1), workgroupsize = 256)
    KernelAbstractions.synchronize(backend)
end

