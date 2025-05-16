"""
    get_edges(X::AbstractMatrix{T}; feature_names, nbins, rng=Random.TaskLocalRNG()) where {T}
    get_edges(df; feature_names, nbins, rng=Random.TaskLocalRNG())

Get the histogram breaking points of the feature data.
"""
function get_edges(X::AbstractMatrix{T}; nbins, rng=Random.MersenneTwister(), kwargs...) where {T}
    @assert T <: Real
    nobs = min(size(X, 1), 1000 * nbins)
    idx = sample(rng, 1:size(X, 1), nobs, replace=false, ordered=true)
    nfeats = size(X, 2)
    edges = Vector{Vector{T}}(undef, nfeats)
    featbins = Vector{UInt8}(undef, nfeats)
    feattypes = Vector{Bool}(undef, nfeats)
    @threads for j in 1:size(X, 2)
        edges[j] = quantile(view(X, idx, j), (1:nbins-1) / nbins)
        if length(edges[j]) == 1
            edges[j] = [minimum(view(X, idx, j))]
        end
        featbins[j] = length(edges[j]) + 1
        feattypes[j] = true
    end
    return edges, featbins, feattypes
end

function get_edges(df; feature_names, nbins, rng=Random.MersenneTwister(), kwargs...)
    _nobs = Tables.DataAPI.nrow(df)
    nobs = min(_nobs, 1000 * nbins)
    idx = sample(rng, 1:_nobs, nobs, replace=false, ordered=true)
    edges = Vector{Any}([Vector{eltype(Tables.getcolumn(df, col))}() for col in feature_names])
    nfeats = length(feature_names)
    featbins = Vector{UInt8}(undef, nfeats)
    feattypes = Vector{Bool}(undef, nfeats)
    @threads for j in eachindex(feature_names)
        col = view(Tables.getcolumn(df, feature_names[j]), idx)
        if eltype(col) <: Bool
            edges[j] = [false, true]
            featbins[j] = 2
            feattypes[j] = false
        elseif eltype(col) <: CategoricalValue
            edges[j] = levels(col)
            featbins[j] = length(edges[j])
            feattypes[j] = isordered(col) ? true : false
            @assert featbins[j] <= 255 "Max categorical levels currently limited to 255, $(feature_names[j]) has $(featbins[j])."
        elseif eltype(col) <: Real
            edges[j] = unique(quantile(col, (1:nbins-1) / nbins))
            featbins[j] = length(edges[j]) + 1
            feattypes[j] = true
        else
            @error "Invalid feature eltype: $(feature_names[j]) is $(eltype(col))"
        end
        if length(edges[j]) == 1
            edges[j] = [minimum(col)]
        end
    end
    return edges, featbins, feattypes
end

"""
    binarize(X::AbstractMatrix; feature_names, edges)
    binarize(df; feature_names, edges)

Transform feature data into a UInt8 binarized matrix.
"""
function binarize(X::AbstractMatrix; feature_names, edges)
    x_bin = zeros(UInt8, size(X))
    @threads for j in axes(X, 2)
        x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end
    return x_bin
end

function binarize(df; feature_names, edges)
    nobs = length(Tables.getcolumn(df, 1))
    x_bin = zeros(UInt8, nobs, length(feature_names))
    @threads for j in eachindex(feature_names)
        col = Tables.getcolumn(df, feature_names[j])
        if eltype(col) <: Bool
            x_bin[:, j] .= col .+ 1
        elseif eltype(col) <: CategoricalValue
            x_bin[:, j] .= levelcode.(col)
        elseif eltype(col) <: Real
            x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), col)
        else
            @error "Invalid feature eltype: $(feature_names[j]) is $(eltype(col))"
        end
    end
    return x_bin
end


function split_set!(
    is_view,
    is,
    left,
    right,
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset,
)

    if length(is_view) < 16_000
        _left, _right = split_set_single!(
            is_view,
            is,
            left,
            right,
            x_bin,
            feat,
            cond_bin,
            feattype,
            offset,
        )
    else
        _left, _right = split_set_threads!(
            is_view,
            is,
            left,
            right,
            x_bin,
            feat,
            cond_bin,
            feattype,
            offset,
        )
    end

    return (_left, _right)
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
    is_view,
    is,
    left,
    right,
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset,
)

    chunk_size = cld(length(is_view), Threads.nthreads())
    nblocks = cld(length(is_view), chunk_size)

    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        lefts[bid], rights[bid] = split_set_chunk!(
            left,
            right,
            is_view,
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

    @threads for bid = 1:nblocks
        split_views_kernel!(
            is,
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
        view(is, offset+1:offset+sum_lefts),
        view(is, offset+sum_lefts+1:offset+length(is_view)),
    )
end

function split_set_single!(
    is_view,
    is,
    left,
    right,
    x_bin,
    feat,
    cond_bin,
    feattype,
    offset,
)
    count_left, count_right = 0, 0

    @inbounds for i in is_view
        cond = feattype ? x_bin[i, feat] <= cond_bin : x_bin[i, feat] == cond_bin
        if cond
            count_left += 1
            left[count_left] = i
        else
            count_right += 1
            right[count_right] = i
        end
    end

    @inbounds for i in 1:count_left
        is[offset+i] = left[i]
    end
    @inbounds for i in 1:count_right
        is[offset+count_left+i] = right[i]
    end

    return (
        view(is, offset+1:offset+count_left),
        view(is, offset+count_left+1:offset+length(is_view)),
    )
end

"""
    update_hist!
        GradientRegression
"""
function update_hist!(
    ::Type{L},
    hist::Array,
    ∇::Matrix,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:GradientRegression}
    hist .= 0
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
    hist::Array,
    ∇::Matrix,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L<:MLE2P}
    hist .= 0
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
        
Generic fallback - Softmax
"""
function update_hist!(
    ::Type{L},
    hist::Array,
    ∇::Matrix,
    x_bin::Matrix,
    is::AbstractVector,
    js::AbstractVector,
) where {L}
    hist .= 0
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
    get_best_split(
        ::Type{L},
        node::TrainNode,
        js,
        params::EvoTypes,
        feattypes::Vector{Bool},
        monotone_constraints,
    )

Generic fallback
"""
function get_best_split(
    ::Type{L},
    node::TrainNode,
    js,
    params::EvoTypes,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L<:LossType}

    h = view(node.h, :, :, js)
    hL = view(node.hL, :, :, js)
    hR = view(node.hR, :, :, js)
    constraints = view(monotone_constraints, js)
    num_flags = view(feattypes, js)
    ∑ = node.∑
    nbins = size(h, 2)
    hL .= h

    best_gain = params.gamma
    best_feat = zero(Int)
    best_bin = zero(Int)

    @inbounds for j in axes(h, 3) # loop over features
        constraint = constraints[j]
        if num_flags[j]
            cumsum!(view(hL, :, :, j), view(hL, :, :, j); dims=2)
            view(hR, :, :, j) .= view(hL, :, nbins, j) .- view(hL, :, :, j)
        else
            view(hR, :, :, j) .= ∑ .- view(hL, :, :, j)
        end
        @inbounds for bin in axes(h, 2) # loop over bins
            if hL[end, bin, j] > params.min_weight && hR[end, bin, j] > params.min_weight
                if constraint != 0
                    predL = pred_scalar(view(hL, :, bin, j), L, params)
                    predR = pred_scalar(view(hR, :, bin, j), L, params)
                end
                if (constraint == 0) ||
                   (constraint == -1 && predL > predR) ||
                   (constraint == 1 && predL < predR)

                    gain = get_gain(L, params, ∑, view(hL, :, bin, j), view(hR, :, bin, j))

                    if gain > best_gain
                        best_gain = gain
                        best_feat = js[j]
                        best_bin = bin
                    end
                end
            end
        end
    end

    return (best_gain, best_feat, best_bin)
end


"""
    update_gains!(
        ::Type{L},
        node::TrainNode,
        js,
        params::EvoTypes,
        feattypes::Vector{Bool},
        monotone_constraints,
    )
"""
function update_gains!(
    ::Type{L},
    node::TrainNode,
    js,
    params::EvoTypes,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L<:LossType}

    h = view(node.h, :, :, js)
    hL = view(node.hL, :, :, js)
    hR = view(node.hR, :, :, js)
    gains = view(node.gains, :, js)
    constraints = view(monotone_constraints, js)
    num_flags = view(feattypes, js)
    ∑ = node.∑
    nbins = size(h, 2)

    gains .= 0 # initialization on demand (rather than at start of tree) 
    hL .= h

    @inbounds for j in axes(h, 3)
        constraint = constraints[j]
        if num_flags[j]
            cumsum!(view(hL, :, :, j), view(hL, :, :, j); dims=2)
            view(hR, :, :, j) .= view(hL, :, nbins, j) .- view(hL, :, :, j)
        else
            view(hR, :, :, j) .= ∑ .- view(hL, :, :, j)
        end
        @inbounds for bin in axes(h, 2)
            if hL[end, bin, j] > params.min_weight && hR[end, bin, j] > params.min_weight
                if constraint != 0
                    predL = pred_scalar(view(hL, :, bin, j), L, params)
                    predR = pred_scalar(view(hR, :, bin, j), L, params)
                end
                if (constraint == 0) ||
                   (constraint == -1 && predL > predR) ||
                   (constraint == 1 && predL < predR)

                    gains[bin, j] = get_gain(L, params, ∑, view(hL, :, bin, j), view(hR, :, bin, j))

                end
            end
        end
    end

    return nothing
end
