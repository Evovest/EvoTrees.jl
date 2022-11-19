#############################################
# Get the braking points
#############################################
function get_edges(X::AbstractMatrix{T}, nbins) where {T}
    Random.seed!(123)
    nobs = min(size(X, 1), 1000 * nbins)
    obs = rand(1:size(X, 1), nobs)
    edges = Vector{Vector{T}}(undef, size(X, 2))
    @threads for i = 1:size(X, 2)
        edges[i] = quantile(view(X, obs, i), (1:nbins) / nbins)
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, obs, i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    x_bin = zeros(UInt8, size(X))
    @threads for i = 1:size(X, 2)
        @inbounds x_bin[:, i] .=
            searchsortedlast.(Ref(edges[i][1:end-1]), view(X, :, i)) .+ 1
    end
    return x_bin
end

"""
    Multi-threaded split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function split_set_chunk!(
    left,
    right,
    is,
    bid,
    nblocks,
    x_bin,
    feat,
    cond_bin,
    offset,
    chunk_size,
)

    left_count = 0
    right_count = 0
    i = chunk_size * (bid - 1) + 1
    bid == nblocks ? bsize = length(is) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        if x_bin[is[i], feat] <= cond_bin
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = is[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = is[i]
        end
        i += 1
    end
    return left_count, right_count
end

function split_views_kernel!(
    out::Vector{S},
    left::Vector{S},
    right::Vector{S},
    bid,
    offset,
    chunk_size,
    lefts,
    rights,
    sum_lefts,
    cumsum_lefts,
    cumsum_rights,
) where {S}
    iter = 1
    i_max = lefts[bid]
    bid == 1 ? cumsum_left = 0 : cumsum_left = cumsum_lefts[bid-1]
    @inbounds while iter <= i_max
        out[offset+cumsum_left+iter] = left[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end

    iter = 1
    i_max = rights[bid]
    bid == 1 ? cumsum_right = 0 : cumsum_right = cumsum_rights[bid-1]
    @inbounds while iter <= i_max
        out[offset+sum_lefts+cumsum_right+iter] = right[offset+chunk_size*(bid-1)+iter]
        iter += 1
    end
    return nothing
end

function split_set_threads!(
    out,
    left,
    right,
    is,
    x_bin::Matrix{S},
    feat,
    cond_bin,
    offset,
) where {S}

    chunk_size = cld(length(is), min(cld(length(is), 1024), Threads.nthreads()))
    nblocks = cld(length(is), chunk_size)

    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        lefts[bid], rights[bid] = split_set_chunk!(
            left,
            right,
            is,
            bid,
            nblocks,
            x_bin,
            feat,
            cond_bin,
            offset,
            chunk_size,
        )
    end

    sum_lefts = sum(lefts)
    cumsum_lefts = cumsum(lefts)
    cumsum_rights = cumsum(rights)
    @threads for bid = 1:nblocks
        split_views_kernel!(
            out,
            left,
            right,
            bid,
            offset,
            chunk_size,
            lefts,
            rights,
            sum_lefts,
            cumsum_lefts,
            cumsum_rights,
        )
    end

    return (
        view(out, offset+1:offset+sum_lefts),
        view(out, offset+sum_lefts+1:offset+length(is)),
    )
end


"""
    update_hist!
        GradientRegression
"""
function update_hist!(
    ::Type{L},
    hist::Array{T,3},
    ∇::Matrix{T},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:GradientRegression,T}
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[1, i]
            hist[2, bin, j] += ∇[2, i]
            hist[3, bin, j] += ∇[3, i]
        end
    end
    return nothing
end

"""
    update_hist!
        MLE2P
"""
function update_hist!(
    ::Type{L},
    hist::Array{T,3},
    ∇::Matrix{T},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:MLE2P,T}
    @threads for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[1, bin, j] += ∇[1, i]
            hist[2, bin, j] += ∇[2, i]
            hist[3, bin, j] += ∇[3, i]
            hist[4, bin, j] += ∇[4, i]
            hist[5, bin, j] += ∇[5, i]
        end
    end
    return nothing
end

"""
    update_hist!
        Generic fallback
"""
function update_hist!(
    ::Type{L},
    hist::Array{T,3},
    ∇::Matrix{T},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L,T}
    @threads for j in js
        @inbounds for i in is
            bin = x_bin[i, j]
            @inbounds @simd for k in axes(∇, 1)
                hist[k, bin, j] += ∇[k, i]
            end
        end
    end
    return nothing
end


"""
    update_gains!(
        loss::L,
        node::TrainNode{T},
        js::Vector,
        params::EvoTypes, K, monotone_constraints) where {L,T,S}

Generic fallback
"""
function update_gains!(
    node::TrainNode,
    js::Vector,
    params::EvoTypes{L,T},
    K,
    monotone_constraints,
) where {L,T}

    KK = 2 * K + 1
    hL = node.hL
    h = node.h
    hL = node.hL
    hR = node.hR
    gains = node.gains

    @inbounds for j in js
        @inbounds for k = 1:KK
            val = h[k, 1, j]
            hL[k, 1, j] = val
            hR[k, 1, j] = node.∑[k] - val
        end
        @inbounds for bin = 2:params.nbins
            @inbounds for k = 1:KK
                val = h[k, bin, j]
                hL[k, bin, j] = hL[k, bin-1, j] + val
                hR[k, bin, j] = hR[k, bin-1, j] - val
            end
        end
    end

    @inbounds for j in js
        monotone_constraint = monotone_constraints[j]
        @inbounds for bin = 1:params.nbins
            if hL[end, bin, j] > params.min_weight && hR[end, bin, j] > params.min_weight
                if monotone_constraint != 0
                    predL = pred_scalar(view(hL, :, bin, j), params)
                    predR = pred_scalar(view(hR, :, bin, j), params)
                end
                if (monotone_constraint == 0) ||
                   (monotone_constraint == -1 && predL > predR) ||
                   (monotone_constraint == 1 && predL < predR)

                    gains[bin, j] =
                        get_gain(params, view(hL, :, bin, j)) +
                        get_gain(params, view(hR, :, bin, j))
                end
            end
        end
    end

    return nothing
end
