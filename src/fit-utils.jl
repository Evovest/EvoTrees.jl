"""
    get_edges(X::AbstractMatrix{T}; fnames, nbins, rng=Random.TaskLocalRNG()) where {T}
    get_edges(df; fnames, nbins, rng=Random.TaskLocalRNG())

Get the histogram breaking points of the feature data.
"""
function get_edges(X::AbstractMatrix{T}; fnames, nbins, rng=Random.MersenneTwister()) where {T}
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

function get_edges(df; fnames, nbins, rng=Random.MersenneTwister())
    _nobs = length(Tables.getcolumn(df, 1))
    nobs = min(_nobs, 1000 * nbins)
    idx = sample(rng, 1:_nobs, nobs, replace=false, ordered=true)
    edges = Vector{Any}([Vector{eltype(Tables.getcolumn(df, col))}() for col in fnames])
    nfeats = length(fnames)
    featbins = Vector{UInt8}(undef, nfeats)
    feattypes = Vector{Bool}(undef, nfeats)
    @threads for j in eachindex(fnames)
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
    @threads for j in axes(X, 2)
        x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end
    return x_bin
end

function binarize(df; fnames, edges)
    nobs = length(Tables.getcolumn(df, 1))
    x_bin = zeros(UInt8, nobs, length(fnames))
    @threads for j in eachindex(fnames)
        col = Tables.getcolumn(df, fnames[j])
        if eltype(col) <: Bool
            x_bin[:, j] .= col .+ 1
        elseif eltype(col) <: CategoricalValue
            x_bin[:, j] .= levelcode.(col)
        elseif eltype(col) <: Real
            x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), col)
        else
            @error "Invalid feature eltype: $(fnames[j]) is $(eltype(col))"
        end
    end
    return x_bin
end

"""
    Multi-threaded split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function update_nodes_idx_cpu!(ns, is, x_bin, cond_feats, cond_bins, feattypes)
    @threads for i in is
        n = ns[i]
        if n == 0
            ns[i] = 0
        else
            feat = cond_feats[n]
            cbin = cond_bins[n]
            if cbin == 0
                ns[i] = 0
            else
                feattype = feattypes[feat]
                is_left = feattype ? x_bin[i, feat] <= cbin : x_bin[i, feat] == cbin
                ns[i] = n << 1 + !is_left
            end
        end
    end
    return nothing
end

# """
#     update_hist!
#         GradientRegression
# """
# function update_hist!(
#     ::Type{L},
#     hist::Vector{Matrix{Float64}},
#     ∇::Matrix{Float32},
#     x_bin::Matrix,
#     is::AbstractVector,
#     js::AbstractVector,
# ) where {L<:GradientRegression}
#     @threads for j in js
#         @inbounds @simd for i in is
#             bin = x_bin[i, j]
#             hist[j][1, bin] += ∇[1, i]
#             hist[j][2, bin] += ∇[2, i]
#             hist[j][3, bin] += ∇[3, i]
#         end
#     end
#     return nothing
# end

# """
#     update_hist!
#         MLE2P
# """
# function update_hist!(
#     ::Type{L},
#     hist::Vector{Matrix{Float64}},
#     ∇::Matrix{Float32},
#     x_bin::Matrix,
#     is::AbstractVector,
#     js::AbstractVector,
# ) where {L<:MLE2P}
#     @threads for j in js
#         @inbounds @simd for i in is
#             bin = x_bin[i, j]
#             hist[j][1, bin] += ∇[1, i]
#             hist[j][2, bin] += ∇[2, i]
#             hist[j][3, bin] += ∇[3, i]
#             hist[j][4, bin] += ∇[4, i]
#             hist[j][5, bin] += ∇[5, i]
#         end
#     end
#     return nothing
# end

"""
    update_hist!
        Generic fallback - Softmax
"""
# function update_hist!(
#     ::Type{L},
#     hist::Vector{Matrix{Float64}},
#     ∇::Matrix{Float32},
#     x_bin::Matrix,
#     is::AbstractVector,
#     js::AbstractVector,
# ) where {L}
#     @threads for j in js
#         @inbounds for i in is
#             bin = x_bin[i, j]
#             @inbounds @simd for k in axes(∇, 1)
#                 hist[j][k, bin] += ∇[k, i]
#             end
#         end
#     end
#     return nothing
# end

function update_hist!(
    h∇::Array{T,4},
    ∇::Matrix{S},
    x_bin::Matrix{UInt8},
    is::AbstractVector,
    js::AbstractVector,
    ns::AbstractVector,
) where {T,S}
    @threads for j in js
        @inbounds for i in is
            n = ns[i]
            if n != 0
                bin = x_bin[i, j]
                h∇[1, bin, j, n] += ∇[1, i]
                h∇[2, bin, j, n] += ∇[2, i]
                h∇[3, bin, j, n] += ∇[3, i]
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
function update_gains!(gains, h∇L, h∇R, h∇, js, dnodes, lambda)
    @threads for j in js
        @inbounds for n in dnodes
            _gains, _h∇L, _h∇R, _h∇ = view(gains, :, j, n), view(h∇L, :, :, j, n), view(h∇R, :, :, j, n), view(h∇, :, :, j, n)
            cumsum!(_h∇L, _h∇; dims=2)
            _h∇R .= _h∇L
            reverse!(_h∇R; dims=2)
            _h∇R .= view(_h∇R, :, 1:1) .- _h∇L
            _gains .= get_gain.(view(_h∇L, 1, :), view(_h∇L, 2, :), view(_h∇L, 3, :), lambda) .+
                      get_gain.(view(_h∇R, 1, :), view(_h∇R, 2, :), view(_h∇R, 3, :), lambda)

            _gains .*= view(_h∇, 3, :) .!= 0
        end
    end
    return nothing
end

const ϵ::Float64 = eps(eltype(Float64))
get_gain(∇1, ∇2, w, lambda) = ∇1^2 / max(ϵ, (∇2 + lambda * w)) / 2
