"""
    build a single histogram containing all grads and weight information
"""
function hist_kernel!(h::CuDeviceArray{T,3}, δ𝑤::CuDeviceMatrix{T}, xid::CuDeviceMatrix{UInt8}, 
    𝑖::CuDeviceVector{S}, 𝑗::CuDeviceVector{S}) where {T,S}
    
    nbins = size(h, 2)

    it = threadIdx().x
    id, jd = blockDim().x, blockDim().y
    ib, j = blockIdx().x, blockIdx().y
    ig, jg = gridDim().x, gridDim().y
    
    shared = @cuDynamicSharedMem(T, (size(h, 1), size(h, 2)))
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    @inbounds while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        @inbounds if i <= length(𝑖) && j <= length(𝑗)
            i_idx = 𝑖[i]
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 1), δ𝑤[1, i_idx])
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 2), δ𝑤[2, i_idx])
            CUDA.atomic_add!(pointer(shared, (xid[i_idx, 𝑗[j]] - 1) * 3 + 3), δ𝑤[3, i_idx])
        end
        iter += 1
    end
    sync_threads()
    # loop over i blocks
    if it <= nbins
        @inbounds hid = Base._to_linear_index(h, 1, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[1, it])

        @inbounds hid = Base._to_linear_index(h, 2, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[2, it])

        @inbounds hid = Base._to_linear_index(h, 3, it, 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, hid), shared[3, it])
    end
    # end
    sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(
    ::L,
    h::CuArray{T,3}, 
    δ𝑤::CuMatrix{T}, 
    X_bin::CuMatrix{UInt8}, 
    𝑖::CuVector{S}, 
    𝑗::CuVector{S}, K;
    MAX_THREADS=256) where {L <: GradientRegression,T,S}
    
    # fill!(h, 0.0)
    thread_i = min(MAX_THREADS, length(𝑖))
    threads = (thread_i, 1)
    blocks = (8, length(𝑗))
    # blocks = (ceil(Int, length(𝑖) / thread_i), length(𝑗))
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 1) * size(h, 2) hist_kernel!(h, δ𝑤, X_bin, 𝑖, 𝑗)
    return
end

# update the vector of length 𝑖 pointing to associated node id
function update_set_kernel!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    it = threadIdx().x
    ibd = blockDim().x
    ibi = blockIdx().x
    i = it + ibd * (ibi - 1)

    if i <= length(𝑖)
        @inbounds idx = 𝑖[i]
        @inbounds feat = feats[𝑛[idx]]
        @inbounds cond = bins[𝑛[idx]]
        @inbounds if cond == 0
            𝑛[idx] = 0
        elseif X_bin[idx, feat] <= cond
            𝑛[idx] = 𝑛[idx] << 1 
        else
            𝑛[idx] = 𝑛[idx] << 1 + 1
        end
    end
    return nothing
end


"""
    Multi-threads split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function split_set_chunk_gpu!(left, right, block, bid, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights, bsizes)
    left_count = 0
    right_count = 0
    @inbounds for i in eachindex(block)
        @inbounds if X_bin[block[i], feat] <= cond_bin
            left_count += 1
            left[offset + chunk_size * (bid - 1) + left_count] = block[i]
        else
            right_count += 1
            right[offset + chunk_size * (bid - 1) + right_count] = block[i]
        end
    end
    lefts[bid] = left_count
    rights[bid] = right_count
    bsizes[bid] = length(block)
    return nothing
end

function split_set_threads_gpu!(out, left, right, 𝑖, X_bin::Matrix{S}, feat, cond_bin, offset, chunk_size=2^15) where {S}    

    left_count = 0 
    right_count = 0
    iter = Iterators.partition(𝑖, chunk_size)
    nblocks = length(iter)
    lefts = zeros(Int, nblocks)
    rights = zeros(Int, nblocks)
    bsizes = zeros(Int, nblocks)

    @sync for (bid, block) in enumerate(iter)
        Threads.@spawn split_set_chunk_gpu!(left, right, block, bid, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights, bsizes)
    end

    left_sum = sum(lefts)
    left_cum = 0
    right_cum = 0
    @inbounds for bid in 1:nblocks
        view(out, offset + left_cum + 1:offset + left_cum + lefts[bid]) .= view(left, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + lefts[bid])
        view(out, offset + left_sum + right_cum + 1:offset + left_sum + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        left_cum += lefts[bid]
        right_cum += rights[bid]
    end

    return (view(out, offset + 1:offset + sum(lefts)), view(out, offset + sum(lefts)+1:offset + length(𝑖)))
end


function update_set_gpu!(𝑛, 𝑖, X_bin, feats, bins, nbins; MAX_THREADS=512)
    thread_i = min(MAX_THREADS, length(𝑖))
    threads = thread_i
    blocks = length(𝑖) ÷ thread_i + 1
    @cuda blocks = blocks threads = threads update_set_kernel!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    return
end

# split row ids into left and right based on best split condition
# function update_set!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    
#     @inbounds for i in 𝑖
#         feat = feats[𝑛[i]]
#         cond = bins[𝑛[i]]
#         if cond == nbins
#             𝑛[𝑖] = 0
#         elseif X_bin[i, feat] <= cond
#             𝑛[i] = 𝑛[i] << 1 
#         else
#             𝑛[i] = 𝑛[i] << 1 + 1
#         end
#     end
#     return nothing
# end


"""
    update_gains!
        Generic fallback
"""
function update_gains_gpu!(
    loss::L,
    node::TrainNode{T},
    𝑗::AbstractVector{S},
    params::EvoTypes, K) where {L,T,S}

    cumsum!(view(node.hL, :, :, :), view(node.h, :, :, :), dims=2)
    # cumsum!(view(histR, :, :, :, nid), reverse!(view(hist, :, :, :, nid), dims=2), dims=2)
    view(node.hR, :, :, :) .= view(node.hL, :, params.nbins:params.nbins, :) .- view(histL, :, :, :)
    hist_gains_gpu!(loss, gains, node.hL, node.hR, 𝑗, params.nbins, params.λ)

    return nothing
end

function hist_gains_gpu_kernel!(gains::CuDeviceMatrix{T}, hL::CuDeviceArray{T,3}, hR::CuDeviceArray{T,3}, 𝑗::CuDeviceVector{S}, nbins, λ::T) where {T,S}
    
    i = threadIdx().x
    j = 𝑗[blockIdx().x]

    @inbounds if hL[3, i, j] > 1e-5 && hR[3, i, j] > 1e-5
        gains[i, j] = (hL[1, i, j]^2 / (hL[2, i, j] + λ * hL[3, i, j]) + 
            hR[1, i, j]^2 / (hR[2, i, j,] + λ * hR[3, i, j])) / 2
    elseif i == nbins
        gains[i, j] = hL[1, i, j]^2 / (hL[2, i, j] + λ * hL[3, i, j]) / 2 
    end
    return nothing
end

function hist_gains_gpu!(loss::L, gains::CuMatrix{T}, hL::CuArray{T,3}, hR::CuArray{T,3}, 𝑗::CuVector{S}, nbins, λ::T; MAX_THREADS=512) where {L <: GradientRegression,T,S}
    thread_i = min(nbins, MAX_THREADS)
    threads = thread_i
    blocks = length(𝑗)
    @cuda blocks = blocks threads = threads hist_gains_gpu_kernel!(gains, hL, hR, 𝑗, nbins, λ)
    return gains
end
