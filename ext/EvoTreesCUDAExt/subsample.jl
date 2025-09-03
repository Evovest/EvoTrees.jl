using KernelAbstractions

@kernel function subsample_step_1_kernel!(is_in, mask, cond::UInt8, counts, chunk_size::Int)
    bid = @index(Global)
    gdim = length(counts)

    i_start = chunk_size * (bid - 1) + 1
    i_stop = bid == gdim ? length(is_in) : i_start + chunk_size - 1
    count = 0

    @inbounds for i = i_start:i_stop
        if mask[i] <= cond
            is_in[i_start+count] = i
            count += 1
        end
    end
    counts[bid] = count
end

@kernel function subsample_step_2_kernel!(is_in, is_out, counts, counts_cum, chunk_size::Int)
    bid = @index(Global)
    count_cum = counts_cum[bid]
    i_start = chunk_size * (bid - 1)
    @inbounds for i = 1:counts[bid]
        is_out[count_cum+i] = is_in[i_start+i]
    end
end

function EvoTrees.subsample(is_in::CuVector, is_out::CuVector, mask::CuVector, rowsample::AbstractFloat, rng)
    backend = KernelAbstractions.get_backend(mask)

    # Fill mask on host for portability, then copy to device
    mask_host = Vector{UInt8}(undef, length(mask))
    Random.rand!(rng, mask_host)
    cond = round(UInt8, 255 * rowsample)
    copyto!(mask, mask_host)

    chunk_size = cld(length(is_in), min(cld(length(is_in), 128), 2048))
    nblocks = cld(length(is_in), chunk_size)
    counts = KernelAbstractions.zeros(backend, Int, nblocks)

    step1! = subsample_step_1_kernel!(backend)
    step1!(is_in, mask, cond, counts, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    counts_host = Array(counts)
    counts_cum_host = cumsum(counts_host) .- counts_host
    counts_cum = KernelAbstractions.zeros(backend, Int, nblocks)
    copyto!(counts_cum, counts_cum_host)

    step2! = subsample_step_2_kernel!(backend)
    step2!(is_in, is_out, counts, counts_cum, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    counts_sum = sum(counts_host)
    if counts_sum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(is_out, 1:counts_sum)
    end
end

