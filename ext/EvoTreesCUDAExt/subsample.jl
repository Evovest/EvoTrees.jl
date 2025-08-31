function get_rand_kernel!(mask)
    tix = threadIdx().x
    bdx = blockDim().x
    bix = blockIdx().x
    gdx = gridDim().x

    i_max = length(mask)
    niter = cld(i_max, bdx * gdx)
    for iter = 1:niter
        i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
        if i <= i_max
            mask[i] = rand(UInt8)
        end
    end
    sync_threads()
end
function get_rand_gpu!(mask)
    threads = (1024,)
    blocks = (256,)
    @cuda threads = threads blocks = blocks get_rand_kernel!(mask)
    CUDA.synchronize()
end

function subsample_step_1_kernel(is_in, mask, cond, counts, chunk_size)

    bid = blockIdx().x
    gdim = gridDim().x

    i_start = chunk_size * (bid - 1) + 1
    i_stop = bid == gdim ? length(is_in) : i_start + chunk_size - 1
    count = 0

    @inbounds for i = i_start:i_stop
        @inbounds if mask[i] <= cond
            is_in[i_start+count] = i
            count += 1
        end
    end
    sync_threads()
    @inbounds counts[bid] = count
    sync_threads()
end

function subsample_step_2_kernel(is_in, is_out, counts, counts_cum, chunk_size)
    bid = blockIdx().x
    count_cum = counts_cum[bid]
    i_start = chunk_size * (bid - 1)
    @inbounds for i = 1:counts[bid]
        is_out[count_cum+i] = is_in[i_start+i]
    end
    sync_threads()
end

function EvoTrees.subsample(is_in::CuVector, is_out::CuVector, mask::CuVector, rowsample::AbstractFloat, rng)
    get_rand_gpu!(mask)
    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(is_in), min(cld(length(is_in), 128), 2048))
    nblocks = cld(length(is_in), chunk_size)
    counts = CUDA.zeros(Int, nblocks)

    blocks = (nblocks,)
    threads = (1,)

    @cuda blocks = nblocks threads = 1 subsample_step_1_kernel(
        is_in,
        mask,
        cond,
        counts,
        chunk_size,
    )
    CUDA.synchronize()
    counts_cum = cumsum(counts) - counts
    @cuda blocks = nblocks threads = 1 subsample_step_2_kernel(
        is_in,
        is_out,
        counts,
        counts_cum,
        chunk_size,
    )
    CUDA.synchronize()
    counts_sum = sum(counts)
    if counts_sum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(is_out, 1:counts_sum)
    end
end

