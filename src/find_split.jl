#############################################
# Get the braking points
#############################################
function get_edges(X::AbstractMatrix{T}, nbins) where {T}
    Random.seed!(123)
    nobs = min(size(X, 1), 1000 * nbins)
    obs = rand(1:size(X, 1), nobs)
    edges = Vector{Vector{T}}(undef, size(X, 2))
    @threads for i = 1:size(X, 2)
        edges[i] = quantile(view(X, obs, i), (1:nbins) / nbins)
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, obs, i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i = 1:size(X, 2)
        X_bin[:, i] .= searchsortedlast.(Ref(edges[i][1:end-1]), view(X, :, i)) .+ 1
    end
    return X_bin
end

"""
    Non Allocating split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
# function split_set!(left::V, right::V, 𝑖, X_bin::Matrix{S}, feat, cond_bin::S, offset) where {S,V}    
#     left_count = 0
#     right_count = 0
#     @inbounds for i in 1:length(𝑖)
#         @inbounds if X_bin[𝑖[i], feat] <= cond_bin
#             left_count += 1
#             left[offset + left_count] = 𝑖[i]
#         else
#             right[offset + length(𝑖) - right_count] = 𝑖[i]
#             right_count += 1
#         end
#     end
#     return (view(left, (offset + 1):(offset + left_count)), view(right, (offset + length(𝑖)):-1:(offset + left_count + 1)))
# end

"""
    Multi-threaded split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function split_set_chunk!(left, right, 𝑖, bid, nblocks, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights)

    left_count = 0
    right_count = 0
    i = chunk_size * (bid - 1) + 1
    bid == nblocks ? bsize = length(𝑖) - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    @inbounds while i <= i_max
        if X_bin[𝑖[i], feat] <= cond_bin
            left_count += 1
            left[offset+chunk_size*(bid-1)+left_count] = 𝑖[i]
        else
            right_count += 1
            right[offset+chunk_size*(bid-1)+right_count] = 𝑖[i]
        end
        i += 1
    end
    @inbounds lefts[bid] = left_count
    @inbounds rights[bid] = right_count
    return nothing
end

function split_views_kernel!(out::Vector{S}, left::Vector{S}, right::Vector{S}, bid, offset, chunk_size, lefts, rights, sum_lefts, cumsum_lefts, cumsum_rights) where {S}
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

function split_set_threads!(out, left, right, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset) where {S}

    # iter = Iterators.partition(𝑖, chunk_size)
    nblocks = ceil(Int, min(length(𝑖) / 1024, Threads.nthreads()))
    chunk_size = floor(Int, length(𝑖) / nblocks)

    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)

    @threads for bid = 1:nblocks
        split_set_chunk!(left, right, 𝑖, bid, nblocks, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights)
    end

    sum_lefts = sum(lefts)
    cumsum_lefts = cumsum(lefts)
    cumsum_rights = cumsum(rights)
    @threads for bid = 1:nblocks
        split_views_kernel!(out, left, right, bid, offset, chunk_size, lefts, rights, sum_lefts, cumsum_lefts, cumsum_rights)
    end

    return (view(out, offset+1:offset+sum_lefts), view(out, offset+sum_lefts+1:offset+length(𝑖)))
end


"""
    update_hist!
        GradientRegression
"""
function update_hist!(
    ::L,
    hist::Vector{Vector{T}},
    δ𝑤::Matrix{T},
    X_bin::Matrix{UInt8},
    𝑖::AbstractVector{S},
    𝑗::AbstractVector{S}, K) where {L<:GradientRegression,T,S}

    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = 3 * X_bin[i, j] - 2
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid+1] += δ𝑤[2, i]
            hist[j][hid+2] += δ𝑤[3, i]
        end
    end
    return nothing
end

"""
    update_hist!
        GaussianRegression
"""
function update_hist!(
    ::L,
    hist::Vector{Vector{T}},
    δ𝑤::Matrix{T},
    X_bin::Matrix{UInt8},
    𝑖::AbstractVector{S},
    𝑗::AbstractVector{S}, K) where {L<:GaussianRegression,T,S}

    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = 5 * X_bin[i, j] - 4
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid+1] += δ𝑤[2, i]
            hist[j][hid+2] += δ𝑤[3, i]
            hist[j][hid+3] += δ𝑤[4, i]
            hist[j][hid+4] += δ𝑤[5, i]
        end
    end
    return nothing
end

"""
    update_hist!
        Generic fallback
"""
function update_hist!(
    ::L,
    hist::Vector{Vector{T}},
    δ𝑤::Matrix{T},
    X_bin::Matrix{UInt8},
    𝑖::AbstractVector{S},
    𝑗::AbstractVector{S}, K) where {L,T,S}

    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = (2 * K + 1) * (X_bin[i, j] - 1)
            for k = 1:(2*K+1)
                hist[j][hid+k] += δ𝑤[k, i]
            end
        end
    end
    return nothing
end


"""
    update_gains!
    GradientRegression
"""
function update_gains!(
    loss::L,
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes, K) where {L<:GradientRegression,T,S}

    @inbounds @threads for j in 𝑗
        node.hL[j][1] = node.h[j][1]
        node.hL[j][2] = node.h[j][2]
        node.hL[j][3] = node.h[j][3]

        node.hR[j][1] = node.∑[1] - node.h[j][1]
        node.hR[j][2] = node.∑[2] - node.h[j][2]
        node.hR[j][3] = node.∑[3] - node.h[j][3]
        @inbounds for bin = 2:params.nbins
            binid = 3 * bin - 2
            node.hL[j][binid] = node.hL[j][binid-3] + node.h[j][binid]
            node.hL[j][binid+1] = node.hL[j][binid-2] + node.h[j][binid+1]
            node.hL[j][binid+2] = node.hL[j][binid-1] + node.h[j][binid+2]

            node.hR[j][binid] = node.hR[j][binid-3] - node.h[j][binid]
            node.hR[j][binid+1] = node.hR[j][binid-2] - node.h[j][binid+1]
            node.hR[j][binid+2] = node.hR[j][binid-1] - node.h[j][binid+2]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.lambda, params.min_weight, K)
    end
    return nothing
end

"""
    update_gains!
    GaussianRegression
"""
function update_gains!(
    loss::L,
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes, K) where {L<:GaussianRegression,T,S}

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
        @inbounds for bin = 2:params.nbins
            binid = 5 * bin - 4
            node.hL[j][binid] = node.hL[j][binid-5] + node.h[j][binid]
            node.hL[j][binid+1] = node.hL[j][binid-4] + node.h[j][binid+1]
            node.hL[j][binid+2] = node.hL[j][binid-3] + node.h[j][binid+2]
            node.hL[j][binid+3] = node.hL[j][binid-2] + node.h[j][binid+3]
            node.hL[j][binid+4] = node.hL[j][binid-1] + node.h[j][binid+4]

            node.hR[j][binid] = node.hR[j][binid-5] - node.h[j][binid]
            node.hR[j][binid+1] = node.hR[j][binid-4] - node.h[j][binid+1]
            node.hR[j][binid+2] = node.hR[j][binid-3] - node.h[j][binid+2]
            node.hR[j][binid+3] = node.hR[j][binid-2] - node.h[j][binid+3]
            node.hR[j][binid+4] = node.hR[j][binid-1] - node.h[j][binid+4]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.lambda, params.min_weight, K)
    end
    return nothing
end


"""
    update_gains!
        Generic fallback
"""
function update_gains!(
    loss::L,
    node::TrainNode{T},
    𝑗::Vector{S},
    params::EvoTypes, K) where {L,T,S}

    KK = 2 * K + 1
    @inbounds @threads for j in 𝑗

        @inbounds for k = 1:KK
            node.hL[j][k] = node.h[j][k]
            node.hR[j][k] = node.∑[k] - node.h[j][k]
        end

        @inbounds for bin = 2:params.nbins
            @inbounds for k = 1:KK
                binid = KK * (bin - 1)
                node.hL[j][binid+k] = node.hL[j][binid-KK+k] + node.h[j][binid+k]
                node.hR[j][binid+k] = node.hR[j][binid-KK+k] - node.h[j][binid+k]
            end
        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.lambda, params.min_weight, K)
    end
    return nothing
end


"""
    hist_gains_cpu!
        GradientRegression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, lambda::T, min_𝑤::T, K) where {L<:GradientRegression,T}
    @inbounds for bin = 1:nbins
        i = 3 * bin - 2
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            gains[bin] = hL[i]^2 / (hL[i+1] + lambda * hL[i+2]) / 2
        elseif hL[i+2] > min_𝑤 && hR[i+2] > min_𝑤
            gains[bin] = (hL[i]^2 / (hL[i+1] + lambda * hL[i+2]) +
                          hR[i]^2 / (hR[i+1] + lambda * hR[i+2])) / 2
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        QuantileRegression/L1Regression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, lambda::T, min_𝑤::T, K) where {L<:Union{QuantileRegression,L1Regression},T}
    @inbounds for bin = 1:nbins
        i = 3 * bin - 2
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            gains[bin] = abs(hL[i])
        elseif hL[i+2] > min_𝑤 && hR[i+2] > min_𝑤
            gains[bin] = abs(hL[i]) + abs(hR[i])
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        GaussianRegression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, lambda::T, min_𝑤::T, K) where {L<:GaussianRegression,T}
    @inbounds for bin = 1:nbins
        i = 5 * bin - 4
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            gains[bin] = (hL[i]^2 / (hL[i+2] + lambda * hL[i+4]) + hL[i+1]^2 / (hL[i+3] + lambda * hL[i+4])) / 2
        elseif hL[i+4] > min_𝑤 && hR[i+4] > min_𝑤
            gains[bin] = (hL[i]^2 / (hL[i+2] + lambda * hL[i+4]) +
                          hR[i]^2 / (hR[i+2] + lambda * hR[i+4])) / 2 +
                         (hL[i+1]^2 / (hL[i+3] + lambda * hL[i+4]) +
                          hR[i+1]^2 / (hR[i+3] + lambda * hR[i+4])) / 2
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        Generic
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, lambda::T, min_𝑤::T, K) where {L,T}
    @inbounds for bin = 1:nbins
        i = (2 * K + 1) * (bin - 1)
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            for k = 1:K
                if k == 1
                    gains[bin] = hL[i+k]^2 / (hL[i+k+K] + lambda * hL[i+2*K+1]) / 2
                else
                    gains[bin] += hL[i+k]^2 / (hL[i+k+K] + lambda * hL[i+2*K+1]) / 2
                end
            end
        elseif hL[i+2*K+1] > min_𝑤 && hR[i+2*K+1] > min_𝑤
            for k = 1:K
                if k == 1
                    gains[bin] = (hL[i+k]^2 / (hL[i+k+K] + lambda * hL[i+2*K+1]) + hR[i+k]^2 / (hR[i+k+K] + lambda * hR[i+2*K+1])) / 2
                else
                    gains[bin] += (hL[i+k]^2 / (hL[i+k+K] + lambda * hL[i+2*K+1]) + hR[i+k]^2 / (hR[i+k+K] + lambda * hR[i+2*K+1])) / 2
                end
            end
        end
    end
    return nothing
end
