"""
    hist_kernel_gauss!
"""
function hist_kernel!(h∇, ∇, x_bin, 𝑖, 𝑗)

    # K = size(h∇, 1)
    tix, tiy, k = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy = blockDim().x, blockDim().y
    bix, biy = blockIdx().x, blockIdx().y
    gdx = gridDim().x

    j = tiy + bdy * (biy - 1)
    if j <= length(𝑗)
        jdx = 𝑗[j]
        i_max = length(𝑖)
        niter = cld(i_max, bdx * gdx)
        iter = 0
        @inbounds for iter = 1:niter
            i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            if i <= length(𝑖)
                @inbounds idx = 𝑖[i]
                @inbounds bin = x_bin[idx, jdx]
                # for k = 1:K
                hid = Base._to_linear_index(h∇, k, bin, jdx)
                CUDA.atomic_add!(pointer(h∇, hid), ∇[k, idx])
                # end
            end
        end
    end
    sync_threads()
    return nothing
end

"""
 dim(x) = dim(i) + 1
"""
@inline index_f(x, i) = x[i]

# x1 = rand(3,2)
# i1 = rand(1:2, 3)
# index_f(1:2, 2)
# index_f(x1, i1)
# index_f.(x1, i1)
# index_f.(x1, i1)

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(h∇, ∇, x_bin, 𝑖, 𝑗; MAX_THREADS=256, MAX_BLOCKS=1024)
    tz = min(64, size(h∇, 1))
    ty = min(length(𝑗), cld(MAX_THREADS, tz))
    tx = min(length(𝑖), cld(MAX_THREADS, tz * ty))
    threads = (tx, ty, tz)
    # @info "threads" threads
    by = cld(length(𝑗), ty)
    bx = min(cld(MAX_BLOCKS, by), cld(length(𝑖), tx))
    blocks = (bx, by, 1)
    # @info "blocks" blocks
    @cuda blocks = blocks threads = threads hist_kernel!(h∇, ∇, x_bin, 𝑖, 𝑗)
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
    𝑖::CuDeviceVector{S},
    X_bin,
    feat,
    cond_bin,
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
    bid == gdim ? bsize = length(𝑖) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        @inbounds if X_bin[𝑖[i], feat] <= cond_bin
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = 𝑖[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = 𝑖[i]
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

    it = threadIdx().x
    bid = blockIdx().x
    gdim = gridDim().x

    # bsize = lefts[bid] + rights[bid]
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

function split_set_threads_gpu!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset)
    𝑖_size = length(𝑖)

    nblocks = ceil(Int, min(length(𝑖) / 128, 2^10))
    chunk_size = floor(Int, length(𝑖) / nblocks)

    lefts = CUDA.zeros(Int, nblocks)
    rights = CUDA.zeros(Int, nblocks)

    # threads = 1
    @cuda blocks = nblocks threads = 1 split_chunk_kernel!(
        left,
        right,
        𝑖,
        X_bin,
        feat,
        cond_bin,
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
        view(out, offset+sum_lefts+1:offset+length(𝑖)),
    )
end


"""
    update_gains!
        GradientRegression
"""
function update_gains!(
    node::TrainNodeGPU,
    𝑗::AbstractVector,
    params::EvoTypes{L,T},
    monotone_constraints;
    MAX_THREADS=512
) where {L,T}

    cumsum!(node.hL, node.h, dims=2)
    node.hR .= view(node.hL, :, params.nbins:params.nbins, :) .- node.hL

    threads = min(params.nbins, MAX_THREADS)
    blocks = length(𝑗)
    @cuda blocks = blocks threads = threads update_gains_kernel!(
        node.gains,
        node.hL,
        node.hR,
        𝑗,
        params.nbins,
        params.lambda,
        params.min_weight,
        monotone_constraints,
    )
    CUDA.synchronize()
    return nothing
end

function update_gains_kernel!(
    gains::CuDeviceMatrix{T},
    hL::CuDeviceArray{T,3},
    hR::CuDeviceArray{T,3},
    𝑗::CuDeviceVector,
    nbins,
    lambda,
    min_weight,
    monotone_constraints,
) where {T}
    bin = threadIdx().x
    j = 𝑗[blockIdx().x]
    monotone_constraint = monotone_constraints[j]
    K = (size(hL, 1) - 1) ÷ 2
    @inbounds for k = 1:K
        if bin == nbins
            gains[bin, j] +=
                hL[k, bin, j]^2 / (hL[k+K, bin, j] + lambda * hL[end, bin, j]) / 2
        elseif hL[end, bin, j] > min_weight && hR[end, bin, j] > min_weight
            if monotone_constraint != 0
                predL = -hL[k, bin, j] / (hL[k+K, bin, j] + lambda * hL[end, bin, j])
                predR = -hR[k, bin, j] / (hR[k+K, bin, j] + lambda * hR[end, bin, j])
            end
            if (monotone_constraint == 0) ||
               (monotone_constraint == -1 && predL > predR) ||
               (monotone_constraint == 1 && predL < predR)
                gains[bin, j] +=
                    (
                        hL[k, bin, j]^2 / (hL[k+K, bin, j] + lambda * hL[end, bin, j]) +
                        hR[k, bin, j]^2 / (hR[k+K, bin, j] + lambda * hR[end, bin, j])
                    ) / 2
            end
        end
    end # loop on K
    sync_threads()
    return nothing
end
