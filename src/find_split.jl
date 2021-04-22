#############################################
# Get the braking points
#############################################
function get_edges(X::AbstractMatrix{T}, nbins=250) where {T}
    edges = Vector{Vector{T}}(undef, size(X, 2))
    @threads for i in 1:size(X, 2)
        edges[i] = quantile(view(X, :, i), (1:nbins) / nbins)
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :, i))]
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
        X_bin[:,i] = searchsortedlast.(Ref(edges[i][1:end - 1]), view(X, :, i)) .+ 1
    end
    X_bin
end

# split row ids into left and right based on best split condition
function split_set!(left, right, 𝑖, X_bin, feat, cond_bin)
    
    left_count = 0 
    right_count = 0

    @inbounds for i in 1:length(𝑖)
        if X_bin[i, feat] <= cond_bin
            left_count += 1
            left[left_count] = 𝑖[i]
        else
            right_count += 1
            right[right_count] = 𝑖[i]
        end
    end
    return (left[1:left_count], right[1:right_count])
    # return (view(left, 1:left_count), view(right, 1:right_count))
end


function update_hist!(
    hist::Vector{Vector{T}}, 
    δ𝑤::Matrix{T}, 
    X_bin::Matrix{UInt8}, 
    𝑖::AbstractVector{S}, 
    𝑗::AbstractVector{S}) where {T,S}
    
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            id = 3 * i - 2
            hid = 3 * X_bin[i,j] - 2
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid + 1] += δ𝑤[2, i]
            hist[j][hid + 2] += δ𝑤[3, i]
        end
    end
    return nothing
end


function update_gains!(
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes, nbins) where {T,S}

    @inbounds @threads for j in 𝑗
        node.hR[j][1] -= node.∑[1]
        node.hR[j][2] -= node.∑[2]
        node.hR[j][3] -= node.∑[3]
        @inbounds for bin in 2:nbins
            binid = 3 * bin - 2
            node.hL[j][binid] = node.hL[j][binid - 3] + node.h[j][binid]
            node.hL[j][binid + 1] = node.hL[j][binid - 2] + node.h[j][binid + 1]
            node.hL[j][binid + 2] = node.hL[j][binid - 1] + node.h[j][binid + 2]

            node.hR[j][binid] = node.hR[j][binid - 3] - node.h[j][binid]
            node.hR[j][binid + 1] = node.hR[j][binid - 2] - node.h[j][binid + 1]
            node.hR[j][binid + 2] = node.hR[j][binid - 1] - node.h[j][binid + 2]

            hist_gains_cpu!(view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ)
        end
    end
    return nothing
end


function hist_gains_cpu!(gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T) where {T}
    @inbounds for bin in 1:nbins
        i = 3 * bin - 2
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if hL[i + 2] > 1e-5 && hR[i + 2] > 1e-5
            @inbounds gains[bin] = (hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) + 
                hR[i]^2 / (hR[i + 1] + λ * hR[i + 2])) / 2
        elseif i == nbins
            @inbounds gains[bin] = hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) / 2 
        end
    end
    return nothing
end
