using KernelAbstractions
using Atomix
using StaticArrays

const MAX_K = 8

@inline get_grad_dims(is_mle2p, K) = is_mle2p ? 5 : (2 * K + 1)
@inline get_workgroup_size(work_size, n_feats=1) = work_size < 256 ? min(128, work_size) : (n_feats > 50 ? min(256, work_size) : min(512, work_size))

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
                is_left = feattypes[feat] ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + T(!is_left)
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
    is_mle2p::Bool,
    n_grad_hess::Int,
    nbins_h::Int,
    n_nodes_h::Int
) where {T}
    gidx = @index(Global, Linear)
    n_feats = length(js)
    n_obs = length(is)
    obs_per_thread = n_obs > 2000000 ? (n_feats > 50 ? 128 : 96) : (n_obs > 500000 ? 80 : 64)

    total_work = cld(n_obs, obs_per_thread) * n_feats
    if gidx <= total_work
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk_idx = (gidx - 1) ÷ n_feats
        feat = js[feat_idx]
        start_idx = obs_chunk_idx * obs_per_thread + 1
        end_idx = min(start_idx + obs_per_thread - 1, n_obs)

        @inbounds for i_obs in start_idx:end_idx
            obs = is[i_obs]
            node = nidx[obs]
            if node > 0 && node <= n_nodes_h
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= nbins_h
                    for k in 1:n_grad_hess
                        Atomix.@atomic h∇[k, bin, feat, node] += ∇[k, obs]
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
    L2::T,
    K::Int,
    is_mae::Bool,
    is_quantile::Bool,
    is_mle2p::Bool,
    n_grad_hess::Int,
    nbins::Int
) where {T}
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            gains[n_idx], bins[n_idx], feats[n_idx] = T(-Inf), Int32(0), Int32(0)
        else
            eps = T(1e-8)

            @inbounds begin
                local_f = js[1]
                for k in 1:n_grad_hess
                    total = zero(T)
                    for b in 1:nbins
                        total += T(h∇[k, b, local_f, node])
                    end
                    nodes_sum[k, node] = total
                end
            end

            w_p = is_mle2p ? nodes_sum[5, node] : nodes_sum[2*K+1, node]
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)

            if w_p >= 2 * min_weight
                parent_gain = zero(T)
                if is_mle2p && K == 2
                    g1, g2 = nodes_sum[1, node], nodes_sum[2, node]
                    h1, h2 = nodes_sum[3, node], nodes_sum[4, node]
                    parent_gain = g1^2 / (h1 + lambda * w_p + L2 + eps) + g2^2 / (h2 + lambda * w_p + L2 + eps)
                else
                    for kk in 1:K
                        g = nodes_sum[kk, node]
                        h = nodes_sum[K+kk, node]
                        parent_gain += g^2 / (h + lambda * w_p + L2 + eps)
                    end
                end

                for j_idx in 1:length(js)
                    f = js[j_idx]
                    constraint = monotone_constraints[f]
                    is_numeric = feattypes[f]
                    s_w = zero(T)
                    cum_g = MVector{MAX_K,T}(ntuple(_ -> zero(T), MAX_K))
                    cum_h = MVector{MAX_K,T}(ntuple(_ -> zero(T), MAX_K))

                    for b in 1:(nbins-1)
                        if is_numeric
                            s_w += is_mle2p ? T(h∇[5, b, f, node]) : T(h∇[2*K+1, b, f, node])
                            for kk in 1:K
                                cum_g[kk] += T(h∇[kk, b, f, node])
                                cum_h[kk] += is_mle2p ? T(h∇[kk+2, b, f, node]) : T(h∇[K+kk, b, f, node])
                            end
                        end

                        l_w = is_numeric ? s_w : (is_mle2p ? T(h∇[5, b, f, node]) : T(h∇[2*K+1, b, f, node]))
                        r_w = w_p - l_w

                        if l_w >= min_weight && r_w >= min_weight
                            left_gain, right_gain = zero(T), zero(T)
                            predL, predR = zero(T), zero(T)

                            for kk in 1:K
                                l_g = is_numeric ? cum_g[kk] : T(h∇[kk, b, f, node])
                                l_h = is_numeric ? cum_h[kk] : (is_mle2p ? T(h∇[kk+2, b, f, node]) : T(h∇[K+kk, b, f, node]))
                                r_g = nodes_sum[kk, node] - l_g
                                r_h = (is_mle2p ? nodes_sum[kk+2, node] : nodes_sum[K+kk, node]) - l_h

                                denomL = l_h + lambda * l_w + L2 + eps
                                denomR = r_h + lambda * r_w + L2 + eps

                                left_gain += l_g^2 / denomL
                                right_gain += r_g^2 / denomR

                                if constraint != 0 && (!is_mle2p || kk == 1)
                                    predL += -l_g / denomL
                                    predR += -r_g / denomR
                                end
                            end

                            constraint_ok = (constraint == 0) || (constraint == -1 && predL > predR) || (constraint == 1 && predL < predR)

                            if constraint_ok
                                g = left_gain + right_gain - parent_gain
                                if g > g_best
                                    g_best, b_best, f_best = g, Int32(b), Int32(f)
                                end
                            end
                        end

                        if is_numeric && (w_p - s_w) < min_weight
                            break
                        end
                    end
                end
                g_best /= 2
            end
            gains[n_idx], bins[n_idx], feats[n_idx] = g_best, b_best, f_best
        end
    end
end

@kernel function count_and_strategy_fused_kernel!(
    node_counts::AbstractVector{Int32},
    build_mask::AbstractVector{UInt8},
    @Const(nidx),
    @Const(is),
    @Const(active_nodes)
)
    gidx = @index(Global)
    @inbounds if gidx <= length(is)
        obs = is[gidx]
        node = nidx[obs]
        if node > 0
            Atomix.@atomic node_counts[node] += Int32(1)
        end
    end
    @synchronize()
    
    if gidx <= length(active_nodes)
        node = active_nodes[gidx]
        if node <= 1
            build_mask[node] = UInt8(1)
        elseif (node & 1) == 0
            sibling = node + 1
            if node_counts[node] <= node_counts[sibling]
                build_mask[node] = UInt8(1)
                build_mask[sibling] = UInt8(0)
            else
                build_mask[node] = UInt8(0)
                build_mask[sibling] = UInt8(1)
            end
        end
    end
end

@kernel function hist_kernel_selective!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(build_mask),
    K::Int,
    is_mle2p::Bool,
    n_grad_hess::Int,
    nbins_h::Int
) where {T}
    gidx = @index(Global)
    @inbounds if gidx <= length(is)
        obs = is[gidx]
        node = nidx[obs]
        
        if node > 0 && node <= length(build_mask) && build_mask[node] > 0
            for j_idx in 1:length(js)
                feat = js[j_idx]
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= nbins_h
                    for k in 1:n_grad_hess
                        Atomix.@atomic h∇[k, bin, feat, node] += ∇[k, obs]
                    end
                end
            end
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(h∇_parent),
    @Const(active_nodes),
    @Const(build_mask),
    n_k::Int,
    n_b::Int,
    n_j::Int
) where {T}
    gidx = @index(Global)
    n_elements = n_k * n_b * n_j
    n_nodes = length(active_nodes)
    
    @inbounds if gidx <= n_nodes * n_elements
        node_idx = (gidx - 1) ÷ n_elements + 1
        elem_idx = (gidx - 1) % n_elements + 1
        
        node = active_nodes[node_idx]
        if node > 1 && build_mask[node] == 0
            parent = node >> 1
            sibling = (node & 1) == 0 ? node + 1 : node - 1
            
            k = (elem_idx - 1) % n_k + 1
            rest = (elem_idx - 1) ÷ n_k
            b = rest % n_b + 1
            j = rest ÷ n_b + 1
            
            h∇[k, b, j, node] = h∇_parent[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

function update_hist_gpu!(
    h∇, h∇_parent,
    gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes,
    nodes_sum_gpu, params, 
    node_counts, build_mask,
    feattypes, monotone_constraints, K;
    is_mae::Bool=false, is_quantile::Bool=false, is_cred::Bool=false, is_mle2p::Bool=false
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if n_active == 0
        return
    end
    
    n_k, n_b, n_j = size(h∇, 1), size(h∇, 2), size(h∇, 3)
    n_nodes_h = size(h∇, 4)
    n_grad_hess = get_grad_dims(is_mle2p, K)
    n_feats = length(js)
    
    h∇ .= 0
    
    if depth == 1
        n_work = cld(length(is), n_feats > 50 ? 96 : 64) * n_feats
        hist_kernel!(backend)(
            h∇, ∇, x_bin, nidx, js, is, K, is_mle2p, n_grad_hess, n_b, n_nodes_h;
            ndrange = n_work, workgroupsize = get_workgroup_size(n_work, n_feats)
        )
        copyto!(h∇_parent, h∇)
    else
        fill!(node_counts, Int32(0))
        fill!(build_mask, UInt8(0))
        
        max_work = max(length(is), length(active_nodes))
        count_and_strategy_fused_kernel!(backend)(
            node_counts, build_mask, nidx, is, active_nodes;
            ndrange = max_work, workgroupsize = get_workgroup_size(max_work)
        )
        
        hist_kernel_selective!(backend)(
            h∇, ∇, x_bin, nidx, js, is, build_mask, K, is_mle2p, n_grad_hess, n_b;
            ndrange = length(is), workgroupsize = get_workgroup_size(length(is))
        )
        
        n_elements = n_k * n_b * n_j
        subtract_hist_kernel!(backend)(
            h∇, h∇_parent, active_nodes, build_mask, n_k, n_b, n_j;
            ndrange = n_active * n_elements,
            workgroupsize = get_workgroup_size(n_active * n_elements)
        )
        
        copyto!(h∇_parent, h∇)
    end
    
    find_best_split_from_hist_kernel!(backend)(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        feattypes, monotone_constraints,
        eltype(gains)(params.lambda),
        eltype(gains)(params.min_weight),
        eltype(gains)(params.L2),
        K, is_mae, is_quantile, is_mle2p, n_grad_hess, n_b;
        ndrange = n_active, workgroupsize = get_workgroup_size(n_active)
    )
    
    KernelAbstractions.synchronize(backend)
end

