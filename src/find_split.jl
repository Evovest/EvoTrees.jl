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


function find_split_turbo!(bins::Vector{BitSet}, X_bin, δ::Vector{S}, δ²::Vector{S}, 𝑤::Vector{S}, ∑δ::S, ∑δ²::S, ∑𝑤::S, params::EvoTreeRegressor, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    info.gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

    track.∑δL = zero(S)
    track.∑δ²L = zero(S)
    track.∑𝑤L = zero(S)
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    hist_δ = zeros(Float64, length(bins))
    hist_δ² = zeros(Float64, length(bins))
    hist_𝑤 = zeros(Float64, length(bins))

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

        if track.gain > info.gain && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight
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

function find_split_bitset!(bins::Vector{BitSet}, δ::Vector{S}, δ²::Vector{S}, 𝑤::Vector{S}, ∑δ::S, ∑δ²::S, ∑𝑤::S, params::EvoTreeRegressor, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    info.gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

    track.∑δL = zero(S)
    track.∑δ²L = zero(S)
    track.∑𝑤L = zero(S)
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    hist_δ = zeros(Float64, length(bins))
    hist_δ² = zeros(Float64, length(bins))
    hist_𝑤 = zeros(Float64, length(bins))

    bin = 1
    # build histogram
    @inbounds for i in set
        bin = 1
        # println("i: ", i, "bins[bin]", bins[bin])
        @inbounds while !(i in bins[bin])
            bin += 1
        end
        hist_δ[bin] += δ[i]
        hist_δ²[bin] += δ²[i]
        hist_𝑤[bin] += 𝑤[i]
    end

    @inbounds for bin in 1:(length(bins)-1)
        track.∑δL += hist_δ[bin]
        track.∑δ²L += hist_δ²[bin]
        track.∑𝑤L += hist_𝑤[bin]
        track.∑δR -= hist_δ[bin]
        track.∑δ²R -= hist_δ²[bin]
        track.∑𝑤R -= hist_𝑤[bin]
        update_track!(params.loss, track, params.λ)

        if track.gain > info.gain && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight
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

# find best split on binarized data
function find_split_bin!(x::AbstractArray{T, 1}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, 𝑤::AbstractArray{Float64, 1}, ∑δ, ∑δ², ∑𝑤, params::EvoTreeRegressor, info::SplitInfo, track::SplitTrack, x_edges) where T<:Real

    info.gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑𝑤L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    @inbounds for i in 1:(size(x, 1) - 1)
        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑𝑤L += 𝑤[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]
        track.∑𝑤R -= 𝑤[i]

        @inbounds if x[i] < x[i+1] && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight # check gain only if there's a change in value
            update_track!(params.loss, track, params.λ)
            if track.gain > info.gain
                info.gain = track.gain
                info.gainL = track.gainL
                info.gainR = track.gainR
                info.∑δL = track.∑δL
                info.∑δ²L = track.∑δ²L
                info.∑𝑤L = track.∑𝑤L
                info.∑δR = track.∑δR
                info.∑δ²R = track.∑δ²R
                info.∑𝑤R = track.∑𝑤R
                info.cond = x_edges[x[i]]
                info.𝑖 = i
            end
        end
    end
end
