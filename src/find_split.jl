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
    Multi-threads split_set!
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
            left[offset + chunk_size * (bid - 1) + left_count] = 𝑖[i]
        else
            right_count += 1
            right[offset + chunk_size * (bid - 1) + right_count] = 𝑖[i]
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
    bid == 1 ? cumsum_left = 0 : cumsum_left = cumsum_lefts[bid - 1]
    @inbounds while iter <= i_max
        out[offset + cumsum_left + iter] = left[offset + chunk_size * (bid - 1) + iter]
        iter += 1
    end

    iter = 1
    i_max = rights[bid]
    bid == 1 ? cumsum_right = 0 : cumsum_right = cumsum_rights[bid - 1]
    @inbounds while iter <= i_max
        out[offset + sum_lefts + cumsum_right + iter] = right[offset + chunk_size * (bid - 1) + iter]
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

    @sync for bid in 1:nblocks
        Threads.@spawn split_set_chunk!(left, right, 𝑖, bid, nblocks, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights)
    end

    sum_lefts = sum(lefts)
    cumsum_lefts = cumsum(lefts)
    cumsum_rights = cumsum(rights)
    @sync for bid in 1:nblocks
        Threads.@spawn split_views_kernel!(out, left, right, bid, offset, chunk_size, lefts, rights, sum_lefts, cumsum_lefts, cumsum_rights)
    end

    return (view(out, offset + 1:offset + sum_lefts), view(out, offset + sum_lefts + 1:offset + length(𝑖)))
end

# f should be a function that take an indices_range and returns a tuple of reduction values
#
# parallel_iterate will unzip those tuples into a tuple of arrays of reduction values and return that.
function parallel_iterate(f, count)
    thread_results = Vector{Any}(undef, nthreads())

    @threads for thread_i in 1:nthreads()
        start = div((thread_i-1) * count, nthreads()) + 1
        stop  = div( thread_i    * count, nthreads())
        thread_results[thread_i] = f(start:stop)
    end

    if isa(thread_results[1], Tuple)
        # Mangling so you get a tuple of arrays.
        Tuple(collect.(zip(thread_results...)))
    else
        thread_results
    end
end


# import InteractiveUtils

"""
    update_hist!
        GradientRegression
"""
function update_hist!(
    loss::L,
    hist::Vector{Vector{T}},
    δ𝑤::Matrix{T},
    X_bin::Matrix{UInt8},
    𝑖::AbstractVector{S},
    𝑗::AbstractVector{S}, K) where {L <: GradientRegression,T,S}

    δ𝑤_flat = reshape(δ𝑤, length(δ𝑤))

    parallel_iterate(length(𝑗)) do 𝑗_range
        𝑗_i = 𝑗_range.start
        while 𝑗_i <= 𝑗_range.stop
          if 𝑗_i + 1 <= 𝑗_range.stop
            j1 = 𝑗[𝑗_i]
            j2 = 𝑗[𝑗_i+1]
            update_hist_gradient!(hist[j1], hist[j2], δ𝑤_flat, X_bin, 𝑖, j1, j2)
            𝑗_i += 2
          else
            j1 = 𝑗[𝑗_i]
            update_hist_gradient!(hist[j1], δ𝑤_flat, X_bin, 𝑖, j1)
            𝑗_i += 1
          end
        end
    end
    return nothing
end

function update_hist_gradient!(
    hist1::Vector{Float32},
    hist2::Vector{Float32},
    δ𝑤_flat::Vector{Float32},
    X_bin,
    𝑖,
    j1,
    j2)
    @inbounds for i in 𝑖
        hid1 = 4 * X_bin[i,j1] - 3
        hid2 = 4 * X_bin[i,j2] - 3
        δ𝑤id = 4 * i - 3

        loss_info = SIMD.vloada(SIMD.Vec{4,Float32}, δ𝑤_flat, δ𝑤id)

        hist_bin1 = SIMD.vloada(SIMD.Vec{4,Float32}, hist1, hid1)
        SIMD.vstorea(hist_bin1 + loss_info, hist1, hid1)
        hist_bin2 = SIMD.vloada(SIMD.Vec{4,Float32}, hist2, hid2)
        SIMD.vstorea(hist_bin2 + loss_info, hist2, hid2)
    end
end

function update_hist_gradient!(hist::Vector{Float32}, δ𝑤_flat::Vector{Float32}, X_bin, 𝑖, j)
    @inbounds for i in 𝑖
        hid  = 4 * X_bin[i,j] - 3
        δ𝑤id = 4 * i - 3

        hist_bin  = SIMD.vloada(SIMD.Vec{4,Float32}, hist, hid)
        loss_info = SIMD.vloada(SIMD.Vec{4,Float32}, δ𝑤_flat, δ𝑤id)
        SIMD.vstorea(hist_bin + loss_info, hist, hid)
    end
end

function update_hist_gradient!(hist1::Vector{T}, hist2::Vector{T}, δ𝑤_flat::Vector{T}, X_bin, 𝑖, j1, j2) where T
    @inbounds @simd for i in 𝑖
        hid1 = 4 * X_bin[i,j1] - 3
        hid2 = 4 * X_bin[i,j2] - 3
        δ𝑤id = 4 * i - 3

        # hist_bin  = SIMD.vloada(SIMD.Vec{4,T}, hist, hid)
        # loss_info = SIMD.vloada(SIMD.Vec{4,T}, δ𝑤_flat, δ𝑤id)
        # SIMD.vstorea(hist_bin + loss_info, hist, hid)
        hist1[hid1]     += δ𝑤_flat[δ𝑤id]
        hist2[hid2]     += δ𝑤_flat[δ𝑤id]
        hist1[hid1 + 1] += δ𝑤_flat[δ𝑤id + 1]
        hist2[hid2 + 1] += δ𝑤_flat[δ𝑤id + 1]
        hist1[hid1 + 2] += δ𝑤_flat[δ𝑤id + 2]
        hist2[hid2 + 2] += δ𝑤_flat[δ𝑤id + 2]
    end
end

function update_hist_gradient!(hist::Vector{T}, δ𝑤_flat::Vector{T}, X_bin, 𝑖, j) where T
    @inbounds @simd for i in 𝑖
        hid  = 4 * X_bin[i,j] - 3
        δ𝑤id = 4 * i - 3

        # hist_bin  = SIMD.vloada(SIMD.Vec{4,T}, hist, hid)
        # loss_info = SIMD.vloada(SIMD.Vec{4,T}, δ𝑤_flat, δ𝑤id)
        # SIMD.vstorea(hist_bin + loss_info, hist, hid)
        hist[hid]     += δ𝑤_flat[δ𝑤id]
        hist[hid + 1] += δ𝑤_flat[δ𝑤id + 1]
        hist[hid + 2] += δ𝑤_flat[δ𝑤id + 2]
    end
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
    𝑗::AbstractVector{S}, K) where {L <: GaussianRegression,T,S}

    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = 8 * X_bin[i,j] - 7
            hist[j][hid] += δ𝑤[1, i]
            hist[j][hid + 1] += δ𝑤[2, i]
        hist[j][hid + 2] += δ𝑤[3, i]
            hist[j][hid + 3] += δ𝑤[4, i]
            hist[j][hid + 4] += δ𝑤[5, i]
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

    stride =  Int(ceil((2 * K + 1)/4)*4)

    @inbounds @threads for j in 𝑗
        @inbounds @simd for i in 𝑖
            hid = stride * (X_bin[i,j] - 1)
            for k in 1:(2 * K + 1)
hist[j][hid + k] += δ𝑤[k, i]
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
    params::EvoTypes, K) where {L <: GradientRegression,T,S}

    @inbounds @threads for j in 𝑗
        node.hL[j][1] = node.h[j][1]
        node.hL[j][2] = node.h[j][2]
        node.hL[j][3] = node.h[j][3]

        node.hR[j][1] = node.∑[1] - node.h[j][1]
        node.hR[j][2] = node.∑[2] - node.h[j][2]
            node.hR[j][3] = node.∑[3] - node.h[j][3]
        @inbounds for bin in 2:params.nbins
            binid = 4 * bin - 3

            node.hL[j][binid] = node.hL[j][binid - 4] + node.h[j][binid]
            node.hL[j][binid + 1] = node.hL[j][binid - 3] + node.h[j][binid + 1]
            node.hL[j][binid + 2] = node.hL[j][binid - 2] + node.h[j][binid + 2]

            node.hR[j][binid] = node.hR[j][binid - 4] - node.h[j][binid]
            node.hR[j][binid + 1] = node.hR[j][binid - 3] - node.h[j][binid + 1]
            node.hR[j][binid + 2] = node.hR[j][binid - 2] - node.h[j][binid + 2]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ, params.min_weight, K)
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
    params::EvoTypes, K) where {L <: GaussianRegression,T,S}

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
            binid = 8 * bin - 7
            node.hL[j][binid] = node.hL[j][binid - 8] + node.h[j][binid]
            node.hL[j][binid + 1] = node.hL[j][binid - 7] + node.h[j][binid + 1]
            node.hL[j][binid + 2] = node.hL[j][binid - 6] + node.h[j][binid + 2]
            node.hL[j][binid + 3] = node.hL[j][binid - 5] + node.h[j][binid + 3]
            node.hL[j][binid + 4] = node.hL[j][binid - 4] + node.h[j][binid + 4]

            node.hR[j][binid] = node.hR[j][binid - 8] - node.h[j][binid]
            node.hR[j][binid + 1] = node.hR[j][binid - 7] - node.h[j][binid + 1]
            node.hR[j][binid + 2] = node.hR[j][binid - 6] - node.h[j][binid + 2]
            node.hR[j][binid + 3] = node.hR[j][binid - 5] - node.h[j][binid + 3]
            node.hR[j][binid + 4] = node.hR[j][binid - 4] - node.h[j][binid + 4]

        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ, params.min_weight, K)
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
    stride = Int(ceil(KK/4)*4)

    @inbounds @threads for j in 𝑗

        @inbounds for k in 1:KK
            node.hL[j][k] = node.h[j][k]
            node.hR[j][k] = node.∑[k] - node.h[j][k]
        end

        @inbounds for bin in 2:params.nbins
            binid = stride * (bin - 1)
            @inbounds for k in 1:KK
                node.hL[j][binid + k] = node.hL[j][binid - stride + k] + node.h[j][binid + k]
                node.hR[j][binid + k] = node.hR[j][binid - stride + k] - node.h[j][binid + k]
            end
        end
        hist_gains_cpu!(loss, view(node.gains, :, j), node.hL[j], node.hR[j], params.nbins, params.λ, params.min_weight, K)
    end
    return nothing
end


"""
    hist_gains_cpu!
        GradientRegression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T, min_𝑤::T, K) where {L <: GradientRegression,T}
    @inbounds for bin in 1:nbins
        i = 4 * bin - 3
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            @inbounds gains[bin] = hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) / 2
        elseif hL[i + 2] > min_𝑤 && hR[i + 2] > min_𝑤
            @inbounds gains[bin] = (hL[i]^2 / (hL[i + 1] + λ * hL[i + 2]) +
                hR[i]^2 / (hR[i + 1] + λ * hR[i + 2])) / 2
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        QuantileRegression/L1Regression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T, min_𝑤::T, K) where {L <: Union{QuantileRegression,L1Regression},T}
    @inbounds for bin in 1:nbins
        i = 4 * bin - 3
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            @inbounds gains[bin] = abs(hL[i])
        elseif hL[i + 2] > min_𝑤 && hR[i + 2] > min_𝑤
@inbounds gains[bin] = abs(hL[i]) + abs(hR[i])
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        GaussianRegression
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T, min_𝑤::T, K) where {L <: GaussianRegression,T}
    @inbounds for bin in 1:nbins
        i = 8 * bin - 7
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        @inbounds if bin == nbins
            gains[bin] = (hL[i]^2 / (hL[i + 2] + λ * hL[i + 4]) + hL[i + 1]^2 / (hL[i + 3] + λ * hL[i + 4])) / 2
        elseif hL[i + 4] > min_𝑤 && hR[i + 4] > min_𝑤
            gains[bin] = (hL[i]^2 / (hL[i + 2] + λ * hL[i + 4]) +
                hR[i]^2 / (hR[i + 2] + λ * hR[i + 4])) / 2 +
                (hL[i + 1]^2 / (hL[i + 3] + λ * hL[i + 4]) +
                hR[i + 1]^2 / (hR[i + 3] + λ * hR[i + 4])) / 2
        end
    end
    return nothing
end

"""
    hist_gains_cpu!
        Generic
"""
function hist_gains_cpu!(::L, gains::AbstractVector{T}, hL::Vector{T}, hR::Vector{T}, nbins, λ::T, min_𝑤::T, K) where {L,T}
    stride =  Int(ceil((2 * K + 1)/4)*4)
    @inbounds for bin in 1:nbins
        i = stride * (bin - 1)
        # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
        if bin == nbins
            @inbounds for k in 1:K
                if k == 1
                    gains[bin] = hL[i + k]^2 / (hL[i + k + K] + λ * hL[i + 2 * K + 1]) / 2
                else
                    gains[bin] += hL[i + k]^2 / (hL[i + k + K] + λ * hL[i + 2 * K + 1]) / 2
                end
            end
        elseif hL[i + 2 * K + 1] > min_𝑤 && hR[i + 2 * K + 1] > min_𝑤
            @inbounds for k in 1:K
                if k == 1
                    gains[bin] = (hL[i + k]^2 / (hL[i + k + K] + λ * hL[i + 2 * K + 1]) +  hR[i + k]^2 / (hR[i + k + K] + λ * hR[i + 2 * K + 1])) / 2
                else
                    gains[bin] += (hL[i + k]^2 / (hL[i + k + K] + λ * hL[i + 2 * K + 1]) +  hR[i + k]^2 / (hR[i + k + K] + λ * hR[i + 2 * K + 1])) / 2
                end
            end
        end
    end
    return nothing
end
