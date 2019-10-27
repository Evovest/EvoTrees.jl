#############################################
# Get the braking points
#############################################
function get_edges(X::Matrix{T}, nbins=250) where {T}
    edges = Vector{Vector{T}}(undef, size(X,2))
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


function find_split_static!(hist_δ::Vector{SVector{L,T}}, hist_δ²::Vector{SVector{L,T}}, hist_𝑤::Vector{SVector{1,T}}, bins::Vector{BitSet}, X_bin, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, params::EvoTreeRegressor, info::SplitInfo{L,T,S}, edges::Vector{T}, set::Vector{S}) where {L,T,S}

    # initialize histogram
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    # initialize tracking
    ∑δL = ∑δ * 0
    ∑δ²L = ∑δ² * 0
    ∑𝑤L = ∑𝑤 * 0
    ∑δR = ∑δ
    ∑δ²R = ∑δ²
    ∑𝑤R = ∑𝑤

    # build histogram
    @inbounds for i in set
        hist_δ[X_bin[i]] += δ[i]
        hist_δ²[X_bin[i]] += δ²[i]
        hist_𝑤[X_bin[i]] += 𝑤[i]
    end

    @inbounds for bin in 1:(length(bins)-1)
        ∑δL += hist_δ[bin]
        ∑δ²L += hist_δ²[bin]
        ∑𝑤L += hist_𝑤[bin]
        ∑δR -= hist_δ[bin]
        ∑δ²R -= hist_δ²[bin]
        ∑𝑤R -= hist_𝑤[bin]

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        if gain > info.gain && ∑𝑤L[1] >= params.min_weight && ∑𝑤R[1] >= params.min_weight
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            info.∑δL = ∑δL
            info.∑δ²L = ∑δ²L
            info.∑𝑤L = ∑𝑤L
            info.∑δR = ∑δR
            info.∑δ²R = ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
    return
end



function update_hist!(hist_δ::Vector{Vector{SVector{L,T}}}, hist_δ²::Vector{Vector{SVector{L,T}}}, hist_𝑤::Vector{Vector{SVector{1,T}}}, X_bin, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}, set::Vector{S}, j::Int) where {L,T,S}
    # build histogram
    hist_δ[j] .*= 0.0
    hist_δ²[j] .*= 0.0
    hist_𝑤[j] .*= 0.0
    @inbounds @simd for i in set
        hist_δ[j][X_bin[i,j]] += δ[i]
        hist_δ²[j][X_bin[i,j]] += δ²[i]
        hist_𝑤[j][X_bin[i,j]] += 𝑤[i]
    end
end

function find_split!(hist_δ::Vector{SVector{L,T}}, hist_δ²::Vector{SVector{L,T}}, hist_𝑤::Vector{SVector{1,T}}, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, params::EvoTreeRegressor, info::SplitInfo{L,T,S}, edges::Vector{T}, j::Int) where {L,T,S}

    # initialize tracking
    ∑δL = ∑δ * 0
    ∑δ²L = ∑δ² * 0
    ∑𝑤L = ∑𝑤 * 0
    ∑δR = ∑δ
    ∑δ²R = ∑δ²
    ∑𝑤R = ∑𝑤

    @inbounds for bin in 1:(length(hist_δ)-1)
        ∑δL += hist_δ[bin]
        ∑δ²L += hist_δ²[bin]
        ∑𝑤L += hist_𝑤[bin]
        ∑δR -= hist_δ[bin]
        ∑δ²R -= hist_δ²[bin]
        ∑𝑤R -= hist_𝑤[bin]

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        if gain > info.gain && ∑𝑤L[1] >= params.min_weight && ∑𝑤R[1] >= params.min_weight
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            info.∑δL = ∑δL
            info.∑δ²L = ∑δ²L
            info.∑𝑤L = ∑𝑤L
            info.∑δR = ∑δR
            info.∑δ²R = ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
end


function find_split_narrow!(hist_δ::Vector{SVector{L,T}}, hist_δ²::Vector{SVector{L,T}}, hist_𝑤::Vector{SVector{1,T}}, bins::Vector{BitSet}, X_bin, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, params::EvoTreeRegressor, info::SplitInfo{L,T,S}, edges::Vector{T}, set::Vector{S}) where {L,T,S}

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
    return
end
