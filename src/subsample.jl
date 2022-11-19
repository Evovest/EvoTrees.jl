"""
    get_rand!(mask)

Assign new UInt8 random numbers to mask. Serves as a basis to rowsampling.
"""
function get_rand!(mask)
    @threads for i in eachindex(mask)
        @inbounds mask[i] = rand(UInt8)
    end
end

"""
    subsample(out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat)

Returns a view of selected rows ids.
"""
function subsample(out::AbstractVector, mask::AbstractVector, rowsample::AbstractFloat)
    get_rand!(mask)
    cond = round(UInt8, 255 * rowsample)
    chunk_size = cld(length(out), min(cld(length(out), 1024), Threads.nthreads()))
    nblocks = cld(length(out), chunk_size)
    counts = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        i_start = chunk_size * (bid - 1) + 1
        i_stop = bid == nblocks ? length(out) : i_start + chunk_size - 1
        count = 0
        i = i_start
        for i = i_start:i_stop
            if mask[i] <= cond
                out[i_start+count] = i
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
            out[count_cum+i] = out[i_start+i]
        end
    end
    counts_sum = sum(counts)
    if counts_cum == 0
        @error "no subsample observation - choose larger rowsample"
    else
        return view(out, 1:counts_sum)
    end
end