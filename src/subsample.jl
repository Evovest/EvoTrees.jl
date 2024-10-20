"""
    subsample(out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat)

Returns a view of selected rows ids.
"""
function subsample(is_in::AbstractVector, is_out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat, rng)
    Random.rand!(rng, mask)

    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(is_in), min(cld(length(is_in), 1024), Threads.nthreads()))
    nblocks = cld(length(is_in), chunk_size)
    counts = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1) + 1
        i_stop = bid == nblocks ? length(is_in) : i_start + chunk_size - 1
        count = 0
        i = i_start
        for i = i_start:i_stop
            if mask[i] <= cond
                is_in[i_start+count] = i
                count += 1
            end
        end
        counts[bid] = count
    end
    counts_cum = cumsum(counts) .- counts
    @threads for bid = 1:nblocks
        count_cum = counts_cum[bid]
        i_start = chunk_size * (bid - 1)
        @inbounds for i = 1:counts[bid]
            is_out[count_cum+i] = is_in[i_start+i]
        end
    end
    counts_sum = sum(counts)
    if counts_cum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(is_out, 1:counts_sum)
    end
end


"""
    subsample(out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat)

Returns a view of selected rows ids.
"""
function subsample(
    is_in::AbstractVector, is_out::AbstractVector, mask::AbstractVector,
    is_in_p::AbstractVector, is_out_p::AbstractVector, mask_p::AbstractVector,
    rowsample::AbstractFloat, rng)

    Random.rand!(rng, mask)
    Random.rand!(rng, mask_p)
    cond = round(UInt8, 255 * rowsample)

    chunk_size = cld(length(is_in), min(cld(length(is_in), 1024), Threads.nthreads()))
    nblocks = cld(length(is_in), chunk_size)
    counts = zeros(Int, nblocks)
    counts_p = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1) + 1
        i_stop = bid == nblocks ? length(is_in) : i_start + chunk_size - 1
        count = 0
        count_p = 0
        i = i_start
        for i = i_start:i_stop
            if mask[i] <= cond
                if mask_p[i]
                    is_in_p[i_start+count_p] = i
                    count_p += 1
                else
                    is_in[i_start+count] = i
                    count += 1
                end
            end
        end
        counts[bid] = count
        counts_p[bid] = count_p
    end

    counts_cum = cumsum(counts) .- counts
    counts_p_cum = cumsum(counts_p) .- counts_p

    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1)
        count = counts[bid]
        count_p = counts_p[bid]
        count_cum = counts_cum[bid]
        count_p_cum = counts_p_cum[bid]
        view(is_out, count_cum+1:count_cum+count) .= view(is_in, i_start+1:i_start+count)
        view(is_out_p, count_p_cum+1:count_p_cum+count_p) .= view(is_in_p, i_start+1:i_start+count_p)
    end
    counts_sum = sum(counts)
    counts_p_sum = sum(counts_p)
    if counts_cum == 0 || counts_p_cum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(is_out, 1:counts_sum), view(is_out_p, 1:counts_p_sum)
    end
end