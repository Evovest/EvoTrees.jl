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
function split_set!(left, right, 𝑖, X_bin::Matrix{S}, feat, cond_bin::S) where S
    
    left_init = 0
    right_init = 0

    left_count = 0
    right_count = 0

    @inbounds for i in 1:length(𝑖)
        id = 𝑖[i]
        @inbounds if X_bin[id, feat] <= cond_bin
            left_count += 1
            left[left_init + left_count] = id
        else
            right_count += 1
            right[right_init + right_count] = id
        end
    end
    return (left[1:left_count], right[1:right_count])
    # return (view(left, 1:left_count), view(right, 1:right_count))
    # return nothing
end

# function split_set!(left, right, 𝑖, X_bin, feat, cond_bin)
    
#     left_count = 0 
#     right_count = 0

#     @inbounds for i in 1:length(𝑖)
#         if X_bin[i, feat] <= cond_bin
#             left_count += 1
#             left[left_count] = 𝑖[i]
#         else
#             right_count += 1
#             right[right_count] = 𝑖[i]
#         end
#     end
#     return nothing
#     # return (left[1:left_count], right[1:right_count])
#     # return (view(left, 1:left_count), view(right, 1:right_count))
# end

function update_hist!(
    ::L,
    hist::Vector{Vector{T}}, 
    δ𝑤::Matrix{T}, 
    X_bin::Matrix{UInt8}, 
    𝑖::AbstractVector{S}, 
    𝑗::AbstractVector{S}) where {L <: GradientRegression,T,S}
    
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = 3 * X_bin[i,j] - 2
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid + 1] += δ𝑤[2, i]
            hist[j][hid + 2] += δ𝑤[3, i]
        end
    end
    return nothing
end

function update_hist!(
    ::L,
    hist::Vector{Vector{T}}, 
    δ𝑤::Matrix{T}, 
    X_bin::Matrix{UInt8}, 
    𝑖::AbstractVector{S}, 
    𝑗::AbstractVector{S}) where {L <: GaussianRegression,T,S}
    
    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = 5 * X_bin[i,j] - 4
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid + 1] += δ𝑤[2, i]
            hist[j][hid + 2] += δ𝑤[3, i]
            hist[j][hid + 3] += δ𝑤[4, i]
            hist[j][hid + 4] += δ𝑤[5, i]
        end
    end
    return nothing
end

function update_gains!(
    loss::L,
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes) where {L <: GradientRegression,T,S}

    @inbounds @threads for j in 𝑗
        node.hL[j][1] = node.h[j][1]
        node.hL[j][2] = node.h[j][2]
        node.hL[j][3] = node.h[j][3]

        node.hR[j][1] = node.∑[1] - node.h[j][1]
        node.hR[j][2] = node.∑[2] - node.h[j][2]
        node.hR[j][3] = node.∑[3] - node.h[j][3]
        @inbounds for bin in 2:params.nbins
            binid = 3 * bin - 2
            node.hL[j][binid] = node.hL[j][binid - 3] + node.h[j][binid]
            node.hL[j][binid + 1] = node.hL[j][binid - 2] + node.h[j][binid + 1]
            node.hL[j][binid + 2] = node.hL[j][binid - 1] + node.h[j][binid + 2]

            node.hR[j][binid] = node.hR[j][binid - 3] - node.h[j][binid]
            node.hR[j][binid + 1] = node.hR[j][binid - 2] - node.h[j][binid + 1]
            node.hR[j][binid + 2] = node.hR[j][binid - 1] - node.h[j][binid + 2]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ)
    end
    return nothing
end


function update_gains!(
    loss::L,
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes) where {L <: GaussianRegression,T,S}

    @inbounds @threads for j in 𝑗
        node.hL[j][1] = node.h[j][1]
        node.hL[j][2] = node.h[j][2]
        node.hL[j][3] = node.h[j][3]
        node.hL[j][4] = node.h[j][4]
        node.hL[j][5] = node.h[j][5]

        node.hR[j][1] = node.∑[1] - node.h[j][1]
        node.hR[j][2] = node.∑[2] - node.h[j][2]
        node.hR[j][3] = node.∑[3] - node.h[j][3]
        node.hR[j][4] = node.∑[4] - node.h[j][4]
        node.hR[j][5] = node.∑[5] - node.h[j][5]
        @inbounds for bin in 2:params.nbins
            binid = 5 * bin - 4
            node.hL[j][binid] = node.hL[j][binid - 5] + node.h[j][binid]
            node.hL[j][binid + 1] = node.hL[j][binid - 4] + node.h[j][binid + 1]
            node.hL[j][binid + 2] = node.hL[j][binid - 3] + node.h[j][binid + 2]
            node.hL[j][binid + 3] = node.hL[j][binid - 2] + node.h[j][binid + 3]
            node.hL[j][binid + 4] = node.hL[j][binid - 1] + node.h[j][binid + 4]

            node.hR[j][binid] = node.hR[j][binid - 5] - node.h[j][binid]
            node.hR[j][binid + 1] = node.hR[j][binid - 4] - node.h[j][binid + 1]
            node.hR[j][binid + 2] = node.hR[j][binid - 3] - node.h[j][binid + 2]
            node.hR[j][binid + 2] = node.hR[j][binid - 2] - node.h[j][binid + 3]
            node.hR[j][binid + 2] = node.hR[j][binid - 1] - node.h[j][binid + 4]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ)
    end
    return nothing
end


function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T) where {L <: GradientRegression,T}
    @inbounds for bin in 1:nbins
        i = 3 * bin - 2
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            @inbounds gains[bin] = hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) / 2 
        elseif hL[i + 2] > 1e-5 && hR[i + 2] > 1e-5
            @inbounds gains[bin] = (hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) + 
                hR[i]^2 / (hR[i + 1] + λ * hR[i + 2])) / 2
        end
    end
    return nothing
end

function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T) where {L <: GaussianRegression,T}
    @inbounds for bin in 1:nbins
        i = 5 * bin - 4
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        @inbounds if bin == nbins
            gains[bin] = (hL[i]^2 / (hL[i + 2] + λ * hL[i + 4]) + hL[i + 1]^2 / (hL[i + 3] + λ * hL[i + 4])) / 2
        elseif hL[i + 4] > 1e-5 && hR[i + 4] > 1e-5
            gains[bin] = (hL[i]^2 / (hL[i + 2] + λ * hL[i + 4]) + 
                hR[i]^2 / (hR[i + 2] + λ * hR[i + 4])) / 2 + 
                (hL[i + 1]^2 / (hL[i + 3] + λ * hL[i + 4]) + 
                hR[i + 1]^2 / (hR[i + 3] + λ * hR[i + 4])) / 2
        end
    end
    return nothing
end
