"""
    get_edges(X::AbstractMatrix{T}; fnames, nbins, rng=Random.TaskLocalRNG()) where {T}
    get_edges(df; fnames, nbins, rng=Random.TaskLocalRNG())

Get the braking points of the feature data.
"""
function get_edges(X::AbstractMatrix{T}; fnames, nbins, rng=Random.TaskLocalRNG()) where {T}
    nobs = min(size(X, 1), 1000 * nbins)
    idx = rand(rng, 1:size(X, 1), nobs)
    nfeats = size(X, 2)
    edges = Vector{Vector{T}}(undef, nfeats)
    featbins = Vector{UInt8}(undef, nfeats)
    feattypes = Vector{Bool}(undef, nfeats)
    @threads :static for j in 1:size(X, 2)
        edges[j] = quantile(view(X, idx, j), (1:nbins-1) / nbins)
        if length(edges[j]) == 1
            edges[j] = [minimum(view(X, idx, j))]
        end
        featbins[j] = length(edges[j]) + 1
        feattypes[j] = true
    end
    return edges, featbins, feattypes
end

function get_edges(df; fnames, nbins, rng=Random.TaskLocalRNG())
    _nobs = length(Tables.getcolumn(df, 1))
    nobs = min(_nobs, 1000 * nbins)
    idx = rand(rng, 1:_nobs, nobs)
    edges = Vector{Any}([Vector{eltype(Tables.getcolumn(df, col))}() for col in fnames])
    nfeats = length(fnames)
    featbins = Vector{UInt8}(undef, nfeats)
    feattypes = Vector{Bool}(undef, nfeats)
    @threads :static for j in eachindex(fnames)
        col = view(Tables.getcolumn(df, fnames[j]), idx)
        if eltype(col) <: Bool
            edges[j] = [false, true]
            featbins[j] = 2
            feattypes[j] = false
        elseif eltype(col) <: CategoricalValue
            edges[j] = levels(col)
            featbins[j] = length(edges[j])
            feattypes[j] = isordered(col) ? true : false
            @assert featbins[j] <= 255 "Max categorical levels currently limited to 255, $(fnames[j]) has $(featbins[j])."
        elseif eltype(col) <: Real
            edges[j] = unique(quantile(col, (1:nbins-1) / nbins))
            featbins[j] = length(edges[j]) + 1
            feattypes[j] = true
        else
            @error "Invalid feature eltype: $(fnames[j]) is $(eltype(col))"
        end
        if length(edges[j]) == 1
            edges[j] = [minimum(col)]
        end
    end
    return edges, featbins, feattypes
end

"""
    binarize(X::AbstractMatrix; fnames, edges)
    binarize(df; fnames, edges)

Transform feature data into a UInt8 binarized matrix.
"""
function binarize(X::AbstractMatrix; fnames, edges)
    x_bin = zeros(UInt8, size(X))
    @threads :static for j in axes(X, 2)
        x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end
    return x_bin
end

function binarize(df; fnames, edges)
    nobs = length(Tables.getcolumn(df, 1))
    x_bin = zeros(UInt8, nobs, length(fnames))
    @threads :static for j in eachindex(fnames)
        col = Tables.getcolumn(df, fnames[j])
        if eltype(col) <: Bool
            x_bin[:, j] .= col .+ 1
        elseif eltype(col) <: CategoricalValue
            x_bin[:, j] .= levelcode.(col)
        elseif eltype(col) <: Real
            x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), col)
        end
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
    feattype,
    offset,
    chunk_size,
)

    left_count = 0
    right_count = 0
    i = chunk_size * (bid - 1) + 1
    bid == nblocks ? bsize = length(is) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        cond = feattype ? x_bin[is[i], feat] <= cond_bin : x_bin[is[i], feat] == cond_bin
        if cond
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
    feattype,
    offset,
) where {S}

    chunk_size = cld(length(is), min(cld(length(is), 1024), Threads.nthreads()))
    nblocks = cld(length(is), chunk_size)

    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)

    @threads :static for bid = 1:nblocks
        lefts[bid], rights[bid] = split_set_chunk!(
            left,
            right,
            is,
            bid,
            nblocks,
            x_bin,
            feat,
            cond_bin,
            feattype,
            offset,
            chunk_size,
        )
    end

    sum_lefts = sum(lefts)
    cumsum_lefts = cumsum(lefts)
    cumsum_rights = cumsum(rights)
    @threads :static for bid = 1:nblocks
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
    hist::Vector{Matrix{Float64}},
    ∇::Matrix{Float32},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:GradientRegression}
    @threads :static for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[j][1, bin] += ∇[1, i]
            hist[j][2, bin] += ∇[2, i]
            hist[j][3, bin] += ∇[3, i]
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
    hist::Vector{Matrix{Float64}},
    ∇::Matrix{Float32},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:MLE2P}
    @threads :static for j in js
        @inbounds @simd for i in is
            bin = x_bin[i, j]
            hist[j][1, bin] += ∇[1, i]
            hist[j][2, bin] += ∇[2, i]
            hist[j][3, bin] += ∇[3, i]
            hist[j][4, bin] += ∇[4, i]
            hist[j][5, bin] += ∇[5, i]
        end
    end
    return nothing
end

"""
    update_hist!
        Generic fallback - Softmax
"""
function update_hist!(
    ::Type{L},
    hist::Vector{Matrix{Float64}},
    ∇::Matrix{Float32},
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L}
    @threads :static for j in js
        @inbounds for i in is
            bin = x_bin[i, j]
            @inbounds @simd for k in axes(∇, 1)
                hist[j][k, bin] += ∇[k, i]
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
    js,
    params::EvoTypes{L},
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L}

    h = node.h
    hL = node.hL
    hR = node.hR
    gains = node.gains
    ∑ = node.∑

    @inbounds for j in js
        if feattypes[j]
            cumsum!(hL[j], h[j], dims=2)
            hR[j] .= ∑ .- hL[j]
        else
            hR[j] .= ∑ .- h[j]
            hL[j] .= h[j]
        end
        monotone_constraint = monotone_constraints[j]
        @inbounds for bin in eachindex(gains[j])
            if hL[j][end, bin] > params.min_weight && hR[j][end, bin] > params.min_weight
                if monotone_constraint != 0
                    predL = pred_scalar(view(hL[j], :, bin), params)
                    predR = pred_scalar(view(hR[j], :, bin), params)
                end
                if (monotone_constraint == 0) ||
                   (monotone_constraint == -1 && predL > predR) ||
                   (monotone_constraint == 1 && predL < predR)

                    gains[j][bin] =
                        get_gain(params, view(hL[j], :, bin)) +
                        get_gain(params, view(hR[j], :, bin))
                end
            end
        end
    end
    return nothing
end
