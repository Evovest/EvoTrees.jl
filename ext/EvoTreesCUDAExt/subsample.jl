function get_rand_kernel!(mask_cond)
    tix = threadIdx().x
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(mask_cond)
    niter = cld(i_max, bdx * gdx)
    for iter = 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            mask_cond[i] = rand(UInt8)
        end
    end
    sync_threads()
end
function get_rand_gpu!(mask_cond)
    threads = (1024,)
    blocks = (256,)
    @cuda threads = threads blocks = blocks get_rand_kernel!(mask_cond)
    CUDA.synchronize()
end

function subsample_step_1_kernel(left, mask_cond, cond, counts, chunk_size)

    bid = blockIdx().x
    gdim = gridDim().x

    i_start = chunk_size * (bid - 1) + 1
    i_stop = bid == gdim ? length(left) : i_start + chunk_size - 1
    count = 0

    @inbounds for i = i_start:i_stop
        @inbounds if mask_cond[i] <= cond
            left[i_start+count] = i
            count += 1
        end
    end
    sync_threads()
    @inbounds counts[bid] = count
    sync_threads()
end

function subsample_step_2_kernel(left, is, counts, counts_cum, chunk_size)
    bid = blockIdx().x
    count_cum = counts_cum[bid]
    i_start = chunk_size * (bid - 1)
    @inbounds for i = 1:counts[bid]
        is[count_cum+i] = left[i_start+i]
    end
    sync_threads()
end

function EvoTrees.subsample(left::CuVector, is::CuVector, mask_cond::CuVector, rowsample::AbstractFloat, rng)
    get_rand_gpu!(mask_cond)
    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(left), min(cld(length(left), 128), 2048))
    nblocks = cld(length(left), chunk_size)
    counts = CUDA.zeros(Int, nblocks)

    blocks = (nblocks,)
    threads = (1,)

    @cuda blocks = nblocks threads = 1 subsample_step_1_kernel(
        left,
        mask_cond,
        cond,
        counts,
        chunk_size,
    )
    CUDA.synchronize()
    counts_cum = cumsum(counts) - counts
    @cuda blocks = nblocks threads = 1 subsample_step_2_kernel(
        left,
        is,
        counts,
        counts_cum,
        chunk_size,
    )
    CUDA.synchronize()
    counts_sum = sum(counts)
    if counts_cum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(is, 1:counts_sum)
    end
end
