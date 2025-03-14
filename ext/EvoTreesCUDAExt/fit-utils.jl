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

function update_hist!(h∇_cpu, h∇::CuArray, ∇::CuMatrix, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇, ∇, x_bin, is, js)
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
    h∇ .= 0
    kernel(h∇, ∇, x_bin, is, js; threads, blocks)
    CUDA.synchronize()
    copyto!(h∇_cpu, h∇)
    return nothing
end


function EvoTrees.split_set!(
    is_view,
    is::CuVector,
    left,
    right,
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset,
)
    _left, _right = EvoTrees.split_set_threads!(
        is_view,
        is,
        left,
        right,
        x_bin,
        feat,
        cond_bin,
        feattype,
        offset,
    )
    return (_left, _right)
end

# Multi-threads split_set!
# Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
function split_chunk_kernel!(
    left::CuDeviceVector{S},
    right::CuDeviceVector{S},
    is_view::CuDeviceVector{S},
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
    bid == gdim ? bsize = length(is_view) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        cond = feattype ? x_bin[is_view[i], feat] <= cond_bin : x_bin[is_view[i], feat] == cond_bin
        if cond
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = is_view[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = is_view[i]
        end
        i += 1
    end
    lefts[bid] = left_count
    rights[bid] = right_count
    sync_threads()
    return nothing
end

function EvoTrees.split_views_kernel!(
    is::CuDeviceVector{S},
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
        is[offset+cumsum_left+iter] = left[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end

    iter = 1
    i_max = rights[bid]
    @inbounds while iter <= i_max
        is[offset+sum_lefts+cumsum_right+iter] = right[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end
    sync_threads()
    return nothing
end

function EvoTrees.split_set_threads!(
    is_view,
    is::CuVector,
    left,
    right,
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset
)
    chunk_size = cld(length(is_view), 1024)
    nblocks = cld(length(is_view), chunk_size)
    lefts = CUDA.zeros(Int, nblocks)
    rights = CUDA.zeros(Int, nblocks)

    # threads = 1
    @cuda blocks = nblocks threads = 1 split_chunk_kernel!(
        left,
        right,
        is_view,
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
    @cuda blocks = nblocks threads = 1 EvoTrees.split_views_kernel!(
        is,
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
        view(is, offset+1:offset+sum_lefts),
        view(is, offset+sum_lefts+1:offset+length(is_view)),
    )
end
