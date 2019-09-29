#############################################
# Get the braking points
#############################################
function get_edges(X, nbins=250)
    edges = Vector{Vector}(undef, size(X,2))
    @threads for i in 1:size(X, 2)
        edges[i] = unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:end]
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :,i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i in 1:size(X, 2)
        X_bin[:,i] = searchsortedlast.(Ref(edges[i][1:end-1]), view(X,:,i)) .+ 1
    end
    X_bin
end

function find_bags(x_bin::Vector{T}) where T <: Real
    𝑖 = 1:length(x_bin) |> collect
    bags = [BitSet() for _ in 1:maximum(x_bin)]
    for bag in 1:length(bags)
        bags[bag] = BitSet(𝑖[x_bin .== bag])
    end
    return bags
end

function update_bags!(bins, set)
    for bin in bins
        intersect!(bin, set)
    end
end


function find_split_turbo!(bins::Vector{BitSet}, X_bin, δ::AbstractVecOrMat{S}, δ²::AbstractVecOrMat{S}, 𝑤::Vector{S}, params::EvoTreeRegressor, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    # initialize histogram
    hist_δ = zeros(Float64, length(bins), size(δ,2))
    hist_δ² = zeros(Float64, length(bins), size(δ,2))
    hist_𝑤 = zeros(Float64, length(bins))

    # build histogram
    @inbounds for i in set
        hist_δ[X_bin[i],:] .+= δ[i,:]
        hist_δ²[X_bin[i],:] .+= δ²[i,:]
        hist_𝑤[X_bin[i]] += 𝑤[i]
    end

    @inbounds for bin in 1:(length(bins)-1)
        track.∑δL .+= hist_δ[bin,:]
        track.∑δ²L .+= hist_δ²[bin,:]
        track.∑𝑤L += hist_𝑤[bin]
        track.∑δR .-= hist_δ[bin,:]
        track.∑δ²R .-= hist_δ²[bin,:]
        track.∑𝑤R -= hist_𝑤[bin]
        update_track!(params.loss, track, params.λ)

        if track.gain > info.gain && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight
            info.gain = track.gain
            info.gainL = track.gainL
            info.gainR = track.gainR
            info.∑δL .= track.∑δL
            info.∑δ²L .= track.∑δ²L
            info.∑𝑤L = track.∑𝑤L
            info.∑δR .= track.∑δR
            info.∑δ²R .= track.∑δ²R
            info.∑𝑤R = track.∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
    return
end


function find_split_static!(hist_δ, hist_δ², hist_𝑤, bins::Vector{BitSet}, X_bin, δ, δ², 𝑤, params::EvoTreeRegressor, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    # initialize histogram
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    # build histogram
    @inbounds for i in set
        hist_δ[X_bin[i]] += δ[i]
        hist_δ²[X_bin[i]] += δ²[i]
        hist_𝑤[X_bin[i]] += 𝑤[i]
    end

    @inbounds for bin in 1:(length(bins)-1)
        track.∑δL += hist_δ[bin]
        track.∑δ²L += hist_δ²[bin]
        track.∑𝑤L += hist_𝑤[bin]
        track.∑δR -= hist_δ[bin]
        track.∑δ²R -= hist_δ²[bin]
        track.∑𝑤R -= hist_𝑤[bin]
        update_track!(params.loss, track, params.λ)

        if track.gain > info.gain && track.∑𝑤L[1] >= params.min_weight && track.∑𝑤R[1] >= params.min_weight
            info.gain = track.gain
            info.gainL = track.gainL
            info.gainR = track.gainR
            info.∑δL = track.∑δL
            info.∑δ²L = track.∑δ²L
            info.∑𝑤L = track.∑𝑤L
            info.∑δR = track.∑δR
            info.∑δ²R = track.∑δ²R
            info.∑𝑤R = track.∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
    return
end
