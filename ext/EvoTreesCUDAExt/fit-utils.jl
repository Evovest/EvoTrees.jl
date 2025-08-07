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
            end
        end
    end
    sync_threads()
    return nothing
end

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
    
    @inbounds for node in dnodes
        h∇[:, :, :, node] .= 0
    end
    
    js_dev = KernelAbstractions.adapt(backend, js)
    kernel! = hist_kernel_ka!(backend)
    kernel!(h∇, ∇, x_bin, nidx, js_dev; ndrange=(size(x_bin, 1), length(js)))
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

