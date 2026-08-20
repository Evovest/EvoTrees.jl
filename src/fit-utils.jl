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
        edges[j] = quantile(view(X, idx, j), (1:(nbins-1)) / nbins)
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
            featbins[j] <= nbins || error("
            Max categorical levels is limited to `nbins` ($nbins). Feature $(feature_names[j]) has $(featbins[j]) levels. Consider using larger `nbins`, up to 255.")
        elseif eltype(col) <: Real
            edges[j] = unique(quantile(col, (1:(nbins-1)) / nbins))
            featbins[j] = length(edges[j]) + 1
            feattypes[j] = true
        else
            error("Invalid feature eltype: $(feature_names[j]) is $(eltype(col))")
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
            error("Invalid feature eltype: $(feature_names[j]) is $(eltype(col))")
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
    split_set_chunk!(
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

Multi-threaded split set.
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
        view(is, (offset+1):(offset+sum_lefts)),
        view(is, (offset+sum_lefts+1):(offset+length(is_view))),
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
        view(is, (offset+1):(offset+count_left)),
        view(is, (offset+count_left+1):(offset+length(is_view))),
    )
end

"""
    subtract_hist!(h∇, nodes, js)

`h∇[:, :, j, n] = h∇[:, :, j, n >> 1] - h∇[:, :, j, n ⊻ 1]` for `n ∈ nodes`, `j ∈ js`.
Reshaped so the `(2K+1, nbins)` plane is one contiguous `@simd` run. Backends
add methods on this signature.
"""
function subtract_hist!(h∇::Array, nodes, js)
    h = reshape(h∇, :, size(h∇, 3), size(h∇, 4))
    @threads for n in nodes
        np, ns = n >> 1, n ⊻ 1
        @inbounds for j in js
            @simd for i in axes(h, 1)
                h[i, j, n] = h[i, j, np] - h[i, j, ns]
            end
        end
    end
    return nothing
end

# llvm.prefetch: read, T0, data cache. llvmcall mangles this from Ref{Int8}, not Ptr.
@inline _prefetch(p::Ptr{UInt8}) = ccall("llvm.prefetch", llvmcall, Cvoid,
    (Ref{Int8}, Int32, Int32, Int32), Ptr{Int8}(p), Int32(0), Int32(3), Int32(1))

"""
    update_hist!(nodes, build_nodes, ∇, x_bin_T, js, ::Val{NK})

Build `nodes[n].h` for each `n` in `build_nodes`. Threads over
(node × feature tile); tiles write disjoint `h[:, :, j]` slices.
`NK == 2K+1` is hist width (`Val` so the inner loop unrolls).
"""
function update_hist!(nodes, build_nodes, ∇, x_bin_T, js, ::Val{NK}) where {NK}
    isempty(build_nodes) && return nothing
    nj, n_build = length(js), length(build_nodes)
    tile = cld(nj, max(1, cld(nthreads(), n_build)))
    ntiles = cld(nj, tile)
    @threads for t = 1:(n_build*ntiles)
        ni, ti = fldmod1(t, ntiles)
        node = nodes[build_nodes[ni]]
        jt = view(js, ((ti-1)*tile+1):min(ti*tile, nj))
        @inbounds for j in jt
            @views fill!(node.h[:, :, j], 0)
        end
        _hist!(node.h, ∇, x_bin_T, node.is, jt, Val(NK))
    end
    return nothing
end

"""
    _hist!(h, ∇, x_bin_T, is, js, ::Val{NK})

Add rows `is`, features `js`, into `h` from observation-major `x_bin_T`.
"""
function _hist!(h, ∇::Matrix{T}, x_bin_T::Matrix{UInt8}, is, js, ::Val{NK}) where {T,NK}
    @assert size(∇, 1) == NK
    nfeats = size(x_bin_T, 1)
    gstride = NK * sizeof(T)
    pxb = Ptr{UInt8}(pointer(x_bin_T))
    pgr = Ptr{UInt8}(pointer(∇))
    n = length(is)
    GC.@preserve x_bin_T ∇ begin
        @inbounds for idx in 1:n
            i = Int(is[idx])
            if idx + PREFETCH_ROWS <= n
                ip = Int(is[idx+PREFETCH_ROWS])
                pb = pxb + (ip - 1) * nfeats
                _prefetch(pb)
                nfeats > 64 && _prefetch(pb + 64)
                _prefetch(pgr + (ip - 1) * gstride)
            end
            g = ntuple(k -> ∇[k, i], Val(NK))
            for j in js
                bin = x_bin_T[j, i]
                for k in 1:NK
                    h[k, bin, j] += g[k]
                end
            end
        end
    end
    return nothing
end

"""
    _scan_node!(sink, ::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints)

Walk eligible split bins of `node`. `sink(gain, j, bin)` is called for each
candidate. Right stats are `parent - left`; `node.∑R` is filled only when a
monotone constraint is active.
"""
@inline function _scan_node!(sink::F, ::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints) where {F,L}
    ∑, ∑L, ∑R = node.∑, node.∑L, node.∑R
    NK = length(∑)
    ϵ = eps(Float64)
    lambda, L2, min_weight = params.lambda, params.L2, params.min_weight
    gain_p = node_gain(L, ∑, lambda, L2, ϵ)
    w_p = ∑[NK]
    @inbounds for j in js
        constraint = monotone_constraints[j]
        is_numeric = feattypes[j]
        fill!(∑L, 0)
        b_max = is_numeric ? size(h∇, 2) - 1 : size(h∇, 2)
        for bin in 1:b_max
            _acc_left!(∑L, h∇, j, bin, n, is_numeric)
            w_l = ∑L[NK]
            w_r = w_p - w_l
            if w_l > min_weight && w_r > min_weight
                if constraint != 0
                    @simd for k in 1:NK
                        ∑R[k] = ∑[k] - ∑L[k]
                    end
                    predL = pred_scalar(∑L, L, params)
                    predR = pred_scalar(∑R, L, params)
                end
                if (constraint == 0) ||
                   (constraint == -1 && predL > predR) ||
                   (constraint == 1 && predL < predR)
                    sink(split_gain(L, ∑, ∑L, w_l, w_r, lambda, L2, ϵ) - gain_p, j, bin)
                end
            end
        end
    end
    return nothing
end

"""
    get_best_split(::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints)

Return `(gain, feat, bin)` for the best split of `node`, or `(gamma, 0, 0)` if
none exceeds `params.gamma`.
"""
function get_best_split(::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints) where {L<:LossType}
    best = Ref((Float64(params.gamma), 0, 0))
    _scan_node!(L, h∇, node, n, js, params, feattypes, monotone_constraints) do gain, j, bin
        gain > best[][1] && (best[] = (gain, Int(j), Int(bin)))
    end
    return best[]
end

"""
    update_gains!(::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints)

Write eligible split gains into `node.gains` (oblivious trees).
"""
function update_gains!(::Type{L}, h∇, node, n, js, params, feattypes, monotone_constraints) where {L<:LossType}
    @views node.gains[:, js] .= 0
    _scan_node!(L, h∇, node, n, js, params, feattypes, monotone_constraints) do gain, j, bin
        node.gains[bin, j] = gain
    end
    return nothing
end

"""
    _child_sums!(∑L, h∇, f, bin, n, is_numeric)

Left-child sums from the parent hist at the winning `(feature, bin)`.
Numeric: bins `1:bin`. Categorical: that bin only.
Right child is `parent ∑ - left`.
"""
function _child_sums!(∑L, h∇, f::Integer, bin::Integer, n::Integer, is_numeric::Bool)
    fill!(∑L, 0)
    @inbounds if is_numeric
        for b in 1:bin
            @simd for k in eachindex(∑L)
                ∑L[k] += h∇[k, b, f, n]
            end
        end
    else
        @simd for k in eachindex(∑L)
            ∑L[k] = h∇[k, bin, f, n]
        end
    end
    return nothing
end
