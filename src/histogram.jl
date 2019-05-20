function scan(X, δ, δ², 𝑤, node, perm_ini, params, splits, tracks, X_edges)
    node_size = size(node.𝑖,1)
    @threads for feat in node.𝑗
        # sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = false)
        sortperm!(view(perm_ini, 1:node_size, feat), X[node.𝑖, feat], alg = QuickSort, initialized = false)
        find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat], X_edges[feat])
    end
end


#############################################
# Quantiles with Sets
#############################################
function find_bags(x::AbstractArray{T, 1}) where T<:Real
    vals = sort(unique(x))
    bags = Vector{BitSet}(undef, length(vals))
    for i in 1:length(vals)
        bags[i] = BitSet(findall(x .== vals[i]))
    end
    return bags
end

function find_bags2(bags, x::AbstractArray{T, 1}, edges) where T<:Real
    x_perm = sortperm(x)
    bin = 1
    for i in x_perm
        if bin > length(edges)
            union!(bags[bin], BitSet(i))
        elseif x[i] <= edges[bin]
            union!(bags[bin], BitSet(i))
        else
            bin += 1
            union!(bags[bin], BitSet(i))
        end
    end
    return bags
end

function update_bags!(bins, set)
    for bin in bins
        intersect!(bin, set)
    end
end

function update_bags_intersect(new_bags, bags, set)
    # new_bags = deepcopy(bags)
    for feat in 1:length(bags)
        for bin in 1:length(bags[feat])
            new_bags[feat][bin] = intersect(set, bags[feat][bin])
            # intersect!(new_bags[feat][bin], set, bags[feat][bin])
        end
    end
    nothing
end

function update_bags_setdiff(new_bags, bags, set)
    # new_bags = deepcopy(bags)
    for feat in 1:length(bags)
        for bin in 1:length(bags[feat])
            new_bags[feat][bin] = setdiff(bags[feat][bin], set)
            # new_bags[feat][bin] = intersect(set, bags[feat][bin])
        end
    end
    nothing
end

function intersect_test(bags, 𝑖_set, δ::S, δ²::S) where {T<:Real,S}
    ∑δ = zero(Float64)
    ∑δ² = zero(Float64)
    ∑δR = zero(Float64)
    ∑δ²R = zero(Float64)
    for bag in bags
        intersect(𝑖_set, bag)
        print(length(𝑖_set))
    end
    return ∑δ
end

function find_histogram(bins, δ::Vector{S}, δ²::Vector{S}, 𝑤::Vector{S}, ∑δ::S, ∑δ²::S, ∑𝑤::S, λ::S, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    info.gain = (∑δ ^ 2 / (∑δ² + λ * ∑𝑤)) / 2.0
    # gain = get_gain(∑δ, ∑δ², ∑𝑤, λ)
    # gainL = zero(S)
    # gainR = zero(S)
    # info.gain = gain

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑𝑤L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    # ∑δL = zero(S)
    # ∑δ²L = zero(S)
    # ∑𝑤L = zero(S)
    # ∑δR = ∑δ
    # ∑δ²R = ∑δ²
    # ∑𝑤R = ∑𝑤

    @inbounds for bin in 1:(length(bins)-1)
        @inbounds for i in bins[bin]
            if i in set
                # ∑δL += δ[i]
                # ∑δ²L += δ²[i]
                # ∑𝑤L += 𝑤[i]
                # ∑δR -= δ[i]
                # ∑δ²R -= δ²[i]
                # ∑𝑤R -= 𝑤[i]

                track.∑δL += δ[i]
                track.∑δ²L += δ²[i]
                track.∑𝑤L += 𝑤[i]
                track.∑δR -= δ[i]
                track.∑δ²R -= δ²[i]
                track.∑𝑤R -= 𝑤[i]
            end
        end
        update_track!(track, λ)
        # if gain > info.gain && ∑𝑤R > zero(S)
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
            info.cond = edges[bin]
            info.𝑖 = bin

            # info.gain = gain
            # info.gainL = gainL
            # info.gainR = gainR
            # info.∑δL = ∑δL
            # info.∑δ²L = ∑δ²L
            # info.∑𝑤L = ∑𝑤L
            # info.∑δR = ∑δR
            # info.∑δ²R = ∑δ²R
            # info.∑𝑤R = ∑𝑤R
            # info.cond = edges[bin]
            # info.𝑖 = bin
        end
    end
    return
end
