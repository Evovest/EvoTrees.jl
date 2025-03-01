"""
    subsample(is_in::AbstractVector, is_out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat, rng)

Returns a view of selected rows ids.
"""
# function subsample(is_in::AbstractVector, out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat, rng)
#     Random.rand!(rng, mask)
#     cond = round(UInt8, 255 * rowsample)
#     out .= mask .<= cond
#     return view(is_in, out)
# end

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

