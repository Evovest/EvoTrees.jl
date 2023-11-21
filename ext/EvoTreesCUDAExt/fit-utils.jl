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
