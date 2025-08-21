<<<<<<< HEAD
"""
    hist_kernel!

Perform a single GPU hist per depth.
Use a vector of node indices. 
"""
function hist_kernel_single_v1!(h∇::CuDeviceArray{T,4}, ∇::CuDeviceMatrix{S}, x_bin, is, js, ns) where {T,S}
    tix, tiy, k = threadIdx().z, threadIdx().y, threadIdx().x
    bdx, bdy = blockDim().z, blockDim().y
    bix, biy = blockIdx().z, blockIdx().y
    gdx = gridDim().z

    j = tiy + bdy * (biy - 1)
    if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        @inbounds for iter in 1:niter
            i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            if i <= i_max
                @inbounds idx = is[i]
                @inbounds ndx = ns[idx]
                if ndx != 0
                    @inbounds bin = x_bin[idx, jdx]
                    hid = Base._to_linear_index(h∇, k, bin, jdx, ndx)
                    CUDA.atomic_add!(pointer(h∇, hid), T(∇[k, idx]))
                end
=======
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
>>>>>>> 1f02d92 (Implement GPU histogram training)
            end
        end
    end
end

<<<<<<< HEAD
function update_hist_gpu_single!(h∇, ∇, x_bin, is, js, ns)
    kernel = @cuda launch = false hist_kernel_single_v1!(h∇, ∇, x_bin, is, js, ns)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    k = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = min(64, max(1, min(length(is), fld(max_threads, k * ty))))
    threads = (k, ty, tx)
    max_blocks = min(65535, max_blocks * fld(max_threads, prod(threads)))
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    kernel(h∇, ∇, x_bin, is, js, ns; threads, blocks)
    CUDA.synchronize()
    return nothing
end


"""
    update_gains_gpu!(gains, h∇L_gpu, h∇R_gpu, h∇_gpu, lambda)
"""
function update_gains_gpu!(gains, h∇L_gpu, h∇R_gpu, h∇_gpu, lambda)
    cumsum!(h∇L_gpu, h∇_gpu; dims=2)
    h∇R_gpu .= h∇L_gpu
    reverse!(h∇R_gpu; dims=2)
    h∇R_gpu .= view(h∇R_gpu, :, 1:1, :, :) .- h∇L_gpu
    gains .= get_gain.(view(h∇L_gpu, 1, :, :, :), view(h∇L_gpu, 2, :, :, :), view(h∇L_gpu, 3, :, :, :), lambda) .+
             get_gain.(view(h∇R_gpu, 1, :, :, :), view(h∇R_gpu, 2, :, :, :), view(h∇R_gpu, 3, :, :, :), lambda)

    gains .*= view(h∇_gpu, 3, :, :, :) .> 1
    gains .*= view(h∇L_gpu, 3, :, :, :) .> 1
    gains .*= view(h∇R_gpu, 3, :, :, :) .> 1

    return nothing
end

const ϵ::Float32 = eps(eltype(Float32))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2


"""
    update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
"""
function update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    tix = threadIdx().x
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(is)
    niter = cld(i_max, bdx * gdx)
    i_ini = niter * (tix - 1) + niter * bdx * (bix - 1)

    @inbounds for iter in 1:niter
        i = i_ini + iter
        if i <= i_max
            idx = is[i]
            n = nidx[idx]
            if n == 0
                nidx[idx] = 0
            else
                feat = cond_feats[n]
                bin = cond_bins[n]
                if bin == 0
                    nidx[idx] = 0
                else
                    feattype = feattypes[feat]
                    is_left = feattype ? x_bin[idx, feat] <= bin : x_bin[idx, feat] == bin
                    nidx[idx] = n << 1 + !is_left
                end
            end
        end
    end
    sync_threads()
    return nothing
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
    kernel = @cuda launch = false update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    threads = min(max_threads, length(is))
    blocks = min(max_blocks, cld(length(is), threads))
    # @info "threads blocks" threads blocks
    # @info "min/max nidx[is]" Int(minimum(nidx[is])) Int(maximum(nidx[is]))
    kernel(nidx, is, x_bin, cond_feats, cond_bins, feattypes; threads, blocks)
    CUDA.synchronize()
    # @info "update idx complete"
    return nothing
end
=======
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
) where {T}
    gidx = @index(Global, Linear)
    
    obs_per_thread = 8
    start_idx = (gidx - 1) * obs_per_thread + 1
    end_idx = min(start_idx + obs_per_thread - 1, length(is))
    
    @inbounds for obs_idx in start_idx:end_idx
        if obs_idx <= length(is)
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4)
                grad1 = ∇[1, obs]
                grad2 = ∇[2, obs]
                grad3 = ∇[3, obs]
                
                @inbounds for j_idx in 1:length(js)
        feat = js[j_idx]
                    if feat <= size(h∇, 3)
                        bin = x_bin[obs, feat]
                        if bin > 0 && bin <= size(h∇, 2)
                            Atomix.@atomic h∇[1, bin, feat, node] += grad1
                            Atomix.@atomic h∇[2, bin, feat, node] += grad2
                            Atomix.@atomic h∇[3, bin, feat, node] += grad3
                        end
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
    lambda::T,
    min_weight::T,
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
            f_first = js[1]
            p_g1 = zero(T); p_g2 = zero(T); p_w = zero(T)
            @inbounds for b in 1:nbins
                p_g1 += h∇[1, b, f_first, node]
                p_g2 += h∇[2, b, f_first, node]
                p_w  += h∇[3, b, f_first, node]
            end
            nodes_sum[1, node] = p_g1
            nodes_sum[2, node] = p_g2
            nodes_sum[3, node] = p_w
            
            gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
            
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            @inbounds for j_idx in 1:length(js)
                f = js[j_idx]
                s1 = zero(T); s2 = zero(T); s3 = zero(T)
                @inbounds for b in 1:(nbins - 1)
                    s1 += h∇[1, b, f, node]
                    s2 += h∇[2, b, f, node]
                    s3 += h∇[3, b, f, node]
                    l_w = s3
                    r_w = p_w - l_w
                    if l_w >= min_weight && r_w >= min_weight
                        l_g1 = s1
                        l_g2 = s2
                        r_g1 = p_g1 - l_g1
                        r_g2 = p_g2 - l_g2
                        gain_l = l_g1^2 / (l_g2 + lambda * l_w + T(1e-8))
                        gain_r = r_g1^2 / (r_g2 + lambda * r_w + T(1e-8))
                        g = gain_l + gain_r - gain_p
                        if g > g_best
                            g_best = g
                            b_best = Int32(b)
                            f_best = Int32(f)
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

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if n_active == 0
        return
    end
    
    h∇ .= 0
    
    num_threads = div(length(is), 8) + 1
    hist_kernel! = hist_kernel!(backend)
    hist_kernel!(h∇, ∇, x_bin, nidx, js, is; ndrange = num_threads)
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

>>>>>>> 1f02d92 (Implement GPU histogram training)
