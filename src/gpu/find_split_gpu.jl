"""
    hist_kernel!
"""
function hist_kernel!(h∇, ∇, x_bin, is)
    tix, k = threadIdx().x, threadIdx().y
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(is)
    niter = cld(i_max, bdx * gdx)
    @inbounds for iter in 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            @inbounds idx = is[i]
            @inbounds bin = x_bin[idx]
            hid = Base._to_linear_index(h∇, k, bin)
            CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
        end
    end
    CUDA.sync_threads()
    return nothing
end

function update_hist_gpu!(h, h∇, ∇, x_bin, is, js)
    kernel = @cuda launch = false hist_kernel!(h∇[js[1]], ∇, view(x_bin, :, js[1]), is)
    config = launch_configuration(kernel.fun)
    max_threads = config.threads
    max_blocks = config.blocks ÷ 4
    ty = size(h∇[js[1]], 1)
    tx = max(1, min(length(is), fld(max_threads, ty)))
    threads = (tx, ty, 1)
    bx = min(max_blocks, cld(length(is), tx))
    blocks = (bx, 1, 1)
    for j in js
        h∇[j] .= 0
    end
    for j in js
        # h∇[j] .= 0
        # @async kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
        kernel(h∇[j], ∇, view(x_bin, :, j), is; threads, blocks)
        # @info "h∇ pre" h∇[js[1]]
    end
    # @info "h∇ post1"
    CUDA.synchronize()
    # @info "h∇ post1" h∇[js[1]]
    # @info "h∇ post1" h∇[js[1]]
    for j in js
        # h∇[j]
        # h[j] .= Array(h∇[j])
        copyto!(h[j], h∇[j])
    end
    # @info "h" h[js[1]]
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
        @inbounds if cond
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = is[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = is[i]
        end
        i += 1
    end
    @inbounds lefts[bid] = left_count
    @inbounds rights[bid] = right_count
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


"""
    update_gains!
        GradientRegression
"""
function update_gains!(
    node::TrainNode,
    js,
    params::EvoTypes{L,T},
    feattypes::CuVector{Bool},
    monotone_constraints,
) where {L,T}

    @sync for j in js
        @async if @allowscalar(feattypes[j])
            cumsum!(node.hL[j], node.h[j], dims=2)
            node.hR[j] .= node.∑ .- node.hL[j]
        else
            node.hR[j] .= node.∑ .- node.h[j]
            node.hL[j] .= node.h[j]
        end
    end
    @sync for j in js
        @async @cuda blocks = (1, 1, 1) threads = length(node.gains[j]) update_gains_kernel!(
            node.gains[j],
            node.hL[j],
            node.hR[j],
            params.lambda,
            params.min_weight,
            monotone_constraints[j],
        )
    end
    CUDA.synchronize()
    return nothing
end

function update_gains_kernel!(
    gains::CuDeviceVector{T},
    hL::CuDeviceMatrix{T},
    hR::CuDeviceMatrix{T},
    lambda,
    min_weight,
    monotone_constraint,
) where {T}
    bin = threadIdx().x
    K = (size(hL, 1) - 1) ÷ 2
    @inbounds for k = 1:K
        if hL[end, bin] > min_weight && hR[end, bin] > min_weight
            if monotone_constraint != 0
                predL = -hL[k, bin] / (hL[k+K, bin] + lambda * hL[end, bin])
                predR = -hR[k, bin] / (hR[k+K, bin] + lambda * hR[end, bin])
            end
            if (monotone_constraint == 0) ||
               (monotone_constraint == -1 && predL > predR) ||
               (monotone_constraint == 1 && predL < predR)
                gains[bin] +=
                    (
                        hL[k, bin]^2 / (hL[k+K, bin] + lambda * hL[end, bin]) +
                        hR[k, bin]^2 / (hR[k+K, bin] + lambda * hR[end, bin])
                    ) / 2
            end
        end
    end # loop on K
    return nothing
end

# function update_gains_kernel!(
#     gains::CuDeviceMatrix{T},
#     hL::CuDeviceArray{T,3},
#     hR::CuDeviceArray{T,3},
#     js::CuDeviceVector,
#     nbins,
#     lambda,
#     min_weight,
#     feattypes,
#     monotone_constraints,
# ) where {T}
#     bin = threadIdx().x
#     j = js[blockIdx().x]
#     monotone_constraint = monotone_constraints[j]
#     K = (size(hL, 1) - 1) ÷ 2
#     @inbounds for k = 1:K
#         if bin == nbins
#             gains[bin, j] +=
#                 hL[k, bin, j]^2 / (hL[k+K, bin, j] + lambda * hL[end, bin, j]) / 2
#         elseif hL[end, bin, j] > min_weight && hR[end, bin, j] > min_weight
#             if monotone_constraint != 0
#                 predL = -hL[k, bin, j] / (hL[k+K, bin, j] + lambda * hL[end, bin, j])
#                 predR = -hR[k, bin, j] / (hR[k+K, bin, j] + lambda * hR[end, bin, j])
#             end
#             if (monotone_constraint == 0) ||
#                (monotone_constraint == -1 && predL > predR) ||
#                (monotone_constraint == 1 && predL < predR)
#                 gains[bin, j] +=
#                     (
#                         hL[k, bin, j]^2 / (hL[k+K, bin, j] + lambda * hL[end, bin, j]) +
#                         hR[k, bin, j]^2 / (hR[k+K, bin, j] + lambda * hR[end, bin, j])
#                     ) / 2
#             end
#         end
#     end # loop on K
#     sync_threads()
#     return nothing
# end
