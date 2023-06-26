"""
    hist_kernel!
"""
function hist_kernel!(h∇::CuDeviceArray{T,3}, ∇::CuDeviceMatrix{S}, x_bin, is, js) where {T,S}
    tix, tiy, k = threadIdx().z, threadIdx().y, threadIdx().x
    bdx, bdy = blockDim().z, blockDim().y
    bix, biy = blockIdx().z, blockIdx().y
    gdx = gridDim().z

    j = tiy + bdy * (biy - 1)
    if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        @inbounds for iter = 1:niter
            i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            if i <= i_max
                @inbounds idx = is[i]
                @inbounds bin = x_bin[idx, jdx]
                hid = Base._to_linear_index(h∇, k, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), T(∇[k, idx]))
            end
        end
    end
    sync_threads()
    return nothing
end

function update_hist_gpu!(h, h∇, ∇, x_bin, is, js, jsc)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads ÷ 4
    max_blocks = config.blocks * 4
    k = size(h∇, 1)
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = min(64, max(1, min(length(is), fld(max_threads, k * ty))))
    threads = (k, ty, tx)
    by = cld(length(js), ty)
    bx = min(cld(max_blocks, by), cld(length(is), tx))
    blocks = (1, by, bx)
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    CUDA.@sync for j in jsc
        nbins = size(h[j], 2)
        copyto!(h[j], view(h∇, :, 1:nbins, j))
    end
    return nothing
end

"""
    hist_kernel_vec!
"""
function hist_kernel_vec!(h∇, ∇, x_bin, is)
    tix, k = threadIdx().x, threadIdx().y
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(is)
    niter = cld(i_max, bdx * gdx)
    @inbounds for iter in 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            idx = is[i]
            bin = x_bin[idx]
            hid = Base._to_linear_index(h∇, k, bin)
            CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
        end
    end
    # CUDA.sync_threads()
    return nothing
end
function update_hist_gpu_vec!(h, h∇, ∇, x_bin, is, js::Vector)
    kernel = @cuda launch = false hist_kernel_vec!(h∇[js[1]], ∇, view(x_bin, :, js[1]), is)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks
    @assert size(h∇[js[1]], 1) <= max_threads "number of classes cannot be larger than 31 on GPU"
    ty = min(64, size(h∇[js[1]], 1))
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    # @sync for j in js
    #     @async h∇[j] .= 0
    # end
    for j in js
        h∇[j] .= 0
        h[j] .= 0
    end
    CUDA.synchronize()
    # @info "hist" max_blocks length(is) threads blocks
    @sync for j in js
        @async kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
        # kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
    end
    CUDA.synchronize()
    for j in js
        copyto!(h[j], h∇[j])
    end
    CUDA.synchronize()
    return nothing
end

"""
    Multi-threads split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function split_chunk_kernel!(
    left::CuDeviceVector{S},
    right::CuDeviceVector{S},
    is::CuDeviceVector{S},
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset,
    chunk_size,
    lefts,
    rights,
) where {S}

    it = threadIdx().x
    bid = blockIdx().x
    gdim = gridDim().x

    left_count = 0
    right_count = 0

    i = chunk_size * (bid - 1) + 1
    bid == gdim ? bsize = length(is) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        cond = feattype ? x_bin[is[i], feat] <= cond_bin : x_bin[is[i], feat] == cond_bin
        if cond
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = is[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = is[i]
        end
        i += 1
    end
    lefts[bid] = left_count
    rights[bid] = right_count
    sync_threads()
    return nothing
end

function split_views_kernel!(
    out::CuDeviceVector{S},
    left::CuDeviceVector{S},
    right::CuDeviceVector{S},
    offset,
    chunk_size,
    lefts,
    rights,
    sum_lefts,
    cumsum_lefts,
    cumsum_rights,
) where {S}

    bid = blockIdx().x
    bid == 1 ? cumsum_left = 0 : cumsum_left = cumsum_lefts[bid-1]
    bid == 1 ? cumsum_right = 0 : cumsum_right = cumsum_rights[bid-1]

    iter = 1
    i_max = lefts[bid]
    @inbounds while iter <= i_max
        out[offset+cumsum_left+iter] = left[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end

    iter = 1
    i_max = rights[bid]
    @inbounds while iter <= i_max
        out[offset+sum_lefts+cumsum_right+iter] = right[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end
    sync_threads()
    return nothing
end

function split_set_threads_gpu!(out, left, right, is, x_bin, feat, cond_bin, feattype, offset)
    chunk_size = cld(length(is), min(cld(length(is), 128), 2048))
    nblocks = cld(length(is), chunk_size)
    lefts = CUDA.zeros(Int, nblocks)
    rights = CUDA.zeros(Int, nblocks)

    # threads = 1
    @cuda blocks = nblocks threads = 1 split_chunk_kernel!(
        left,
        right,
        is,
        x_bin,
        feat,
        cond_bin,
        feattype,
        offset,
        chunk_size,
        lefts,
        rights,
    )
    CUDA.synchronize()

    sum_lefts = sum(lefts)
    cumsum_lefts = cumsum(lefts)
    cumsum_rights = cumsum(rights)
    @cuda blocks = nblocks threads = 1 split_views_kernel!(
        out,
        left,
        right,
        offset,
        chunk_size,
        lefts,
        rights,
        sum_lefts,
        cumsum_lefts,
        cumsum_rights,
    )

    CUDA.synchronize()
    return (
        view(out, offset+1:offset+sum_lefts),
        view(out, offset+sum_lefts+1:offset+length(is)),
    )
end
