"""
    build a single histogram containing all grads and weight information
"""
function hist_kernel_grad!(
    h::CuDeviceArray{T,3},
    δ𝑤::CuDeviceMatrix{T},
    xid::CuDeviceMatrix{UInt8},
    𝑖::CuDeviceVector{S},
    𝑗::CuDeviceVector{S},
) where {T,S}

    nbins = size(h, 2)

    it = threadIdx().x
    id, jd = blockDim().x, blockDim().y
    ib, j = blockIdx().x, blockIdx().y
    ig, jg = gridDim().x, gridDim().y

    # shared = @cuDynamicSharedMem(T, (size(h, 1), size(h, 2)))
    shared = CuDynamicSharedArray(T, (size(h, 1), size(h, 2)))
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    @inbounds while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        @inbounds if i <= length(𝑖) && j <= length(𝑗)
            i_idx = 𝑖[i]
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 1), δ𝑤[1, i_idx])
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 2), δ𝑤[2, i_idx])
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 3), δ𝑤[3, i_idx])
        end
        iter += 1
    end
    sync_threads()
    # loop over i blocks
    if it <= nbins
        @inbounds hid = Base._to_linear_index(h, 1, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[1, it])

        @inbounds hid = Base._to_linear_index(h, 2, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[2, it])

        @inbounds hid = Base._to_linear_index(h, 3, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[3, it])
    end
    sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(
    ::Type{L},
    h::CuArray{T,3},
    δ𝑤::CuMatrix{T},
    X_bin::CuMatrix{UInt8},
    𝑖::CuVector{S},
    𝑗::CuVector{S},
    K;
    MAX_THREADS = 256,
) where {L<:GradientRegression,T,S}

    nbins = size(h, 2)
    thread_i = max(nbins, min(MAX_THREADS, length(𝑖)))
    threads = (thread_i, 1)
    blocks = (8, length(𝑗))
    # println("threads: ", threads, " | blocks: ", blocks)
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 1) * nbins hist_kernel_grad!(
        h,
        δ𝑤,
        X_bin,
        𝑖,
        𝑗,
    )
    CUDA.synchronize()
    return nothing
end


"""
    hist_kernel_gauss!
"""
function hist_kernel_gauss!(
    h::CuDeviceArray{T,3},
    δ𝑤::CuDeviceMatrix{T},
    xid::CuDeviceMatrix{UInt8},
    𝑖::CuDeviceVector{S},
    𝑗::CuDeviceVector{S},
) where {T,S}

    nbins = size(h, 2)

    it, k = threadIdx().x, threadIdx().y
    id, jd = blockDim().x, blockDim().y
    ib, j = blockIdx().x, blockIdx().y
    ig, jg = gridDim().x, gridDim().y

    # shared = @cuDynamicSharedMem(T, (size(h, 1), size(h, 2)))
    shared = CuDynamicSharedArray(T, (size(h, 1), size(h, 2)))
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    @inbounds while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        @inbounds if i <= length(𝑖) && j <= length(𝑗)
            i_idx = 𝑖[i]
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 5 + k), δ𝑤[k, i_idx])
        end
        iter += 1
    end
    sync_threads()
    # loop over i blocks
    if it <= nbins
        @inbounds hid = Base._to_linear_index(h, k, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[k, it])
    end
    sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(
    ::Type{L},
    h::CuArray{T,3},
    δ𝑤::CuMatrix{T},
    X_bin::CuMatrix{UInt8},
    𝑖::CuVector{S},
    𝑗::CuVector{S},
    K;
    MAX_THREADS = 128,
) where {L<:MLE2P,T,S}

    nbins = size(h, 2)
    thread_i = max(nbins, min(MAX_THREADS, length(𝑖)))
    threads = (thread_i, 5)
    blocks = (8, length(𝑗))
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 1) * nbins hist_kernel_gauss!(
        h,
        δ𝑤,
        X_bin,
        𝑖,
        𝑗,
    )
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
function update_gains_gpu!(
    node::TrainNodeGPU,
    𝑗::AbstractVector,
    params::EvoTypes{L,T,S},
    K,
    monotone_constraints;
    MAX_THREADS = 512,
) where {L<:GradientRegression,T,S}

    cumsum!(node.hL, node.h, dims = 2)
    node.hR .= view(node.hL, :, params.nbins:params.nbins, :) .- node.hL

    threads = min(params.nbins, MAX_THREADS)
    blocks = length(𝑗)
    @cuda blocks = blocks threads = threads hist_gains_gpu_kernel!(
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

function hist_gains_gpu_kernel!(
    gains::CuDeviceMatrix{T},
    hL::CuDeviceArray{T,3},
    hR::CuDeviceArray{T,3},
    𝑗::CuDeviceVector{S},
    nbins,
    lambda,
    min_weight,
    monotone_constraints,
) where {T,S}

    i = threadIdx().x
    j = 𝑗[blockIdx().x]
    monotone_constraint = monotone_constraints[j]

    if i == nbins
        gains[i, j] = hL[1, i, j]^2 / (hL[2, i, j] + lambda * hL[3, i, j]) / 2
    elseif hL[3, i, j] > min_weight && hR[3, i, j] > min_weight
        if monotone_constraint != 0
            predL = -hL[1, i, j] / (hL[2, i, j] + lambda * hL[3, i, j])
            predR = -hR[1, i, j] / (hR[2, i, j] + lambda * hR[3, i, j])
        end
        if (monotone_constraint == 0) ||
           (monotone_constraint == -1 && predL > predR) ||
           (monotone_constraint == 1 && predL < predR)
            gains[i, j] =
                (
                    hL[1, i, j]^2 / (hL[2, i, j] + lambda * hL[3, i, j]) +
                    hR[1, i, j]^2 / (hR[2, i, j] + lambda * hR[3, i, j])
                ) / 2
        end
    end
    sync_threads()
    return nothing
end


"""
    update_gains!
        MLE2P
"""
function update_gains_gpu!(
    node::TrainNodeGPU,
    𝑗::AbstractVector,
    params::EvoTypes{L,T,S},
    K,
    monotone_constraints;
    MAX_THREADS = 512,
) where {L<:MLE2P,T,S}

    cumsum!(node.hL, node.h, dims = 2)
    node.hR .= view(node.hL, :, params.nbins:params.nbins, :) .- node.hL

    thread_i = min(params.nbins, MAX_THREADS)
    threads = thread_i
    blocks = length(𝑗)
    @cuda blocks = blocks threads = threads hist_gains_gpu_kernel_gauss!(
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

function hist_gains_gpu_kernel_gauss!(
    gains::CuDeviceMatrix{T},
    hL::CuDeviceArray{T,3},
    hR::CuDeviceArray{T,3},
    𝑗::CuDeviceVector{S},
    nbins,
    lambda,
    min_weight,
    monotone_constraints,
) where {T,S}

    i = threadIdx().x
    j = 𝑗[blockIdx().x]
    monotone_constraint = monotone_constraints[j]

    if i == nbins
        gains[i, j] =
            (
                hL[1, i, j]^2 / (hL[3, i, j] + lambda * hL[5, i, j]) +
                hL[2, i, j]^2 / (hL[4, i, j] + lambda * hL[5, i, j])
            ) / 2
    elseif hL[5, i, j] > min_weight && hR[5, i, j] > min_weight
        if monotone_constraint != 0
            predL = -hL[1, i, j] / (hL[3, i, j] + lambda * hL[5, i, j])
            predR = -hR[1, i, j] / (hR[3, i, j] + lambda * hR[5, i, j])
        end
        if (monotone_constraint == 0) ||
           (monotone_constraint == -1 && predL > predR) ||
           (monotone_constraint == 1 && predL < predR)
            gains[i, j] =
                (
                    hL[1, i, j]^2 / (hL[3, i, j] + lambda * hL[5, i, j]) +
                    hR[1, i, j]^2 / (hR[3, i, j] + lambda * hR[5, i, j])
                ) / 2 +
                (
                    hL[2, i, j]^2 / (hL[4, i, j] + lambda * hL[5, i, j]) +
                    hR[2, i, j]^2 / (hR[4, i, j] + lambda * hR[5, i, j])
                ) / 2
        end
    end
    sync_threads()
    return nothing
end
