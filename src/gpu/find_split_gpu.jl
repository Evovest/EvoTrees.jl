"""
    build a single histogram containing all grads and weight information
"""
# GPU - apply along the features axis
# function hist_kernel!(h::CuDeviceArray{T,4}, δ::CuDeviceMatrix{T}, idx::CuDeviceMatrix{UInt8}, 
#     𝑖::CuDeviceVector{S}, 𝑗::CuDeviceVector{S}, 𝑛::CuDeviceVector{S}, L) where {T,S}

#     it = threadIdx().x
#     id = blockDim().x
#     ib = blockIdx().x
#     ig = gridDim().x

#     # k = blockIdx().z
#     # i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

#     j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

#     i_tot = length(𝑖)
#     iter = 0
#     while iter * id * ig < i_tot
#         i = it + id * (ib - 1) + iter * id * ig
#         if i <= length(𝑖) && j <= length(𝑗)
#             n = 𝑛[i]
#             if n > 0
#                 pt = Base._to_linear_index(h, 1, idx[𝑖[i], 𝑗[j]], 𝑗[j], n)
#                 for k in 1:L
                    # CUDA.atomic_add!(pointer(h, pt + k - 1), δ[𝑖[i], k])
#                 end
#             end
#         end
#         iter += 1
#     end
#     return nothing
# end


function hist_kernel!(h::CuDeviceArray{T,4}, δ::CuDeviceMatrix{T}, xid::CuDeviceMatrix{UInt8}, 
    𝑖::CuDeviceVector{S}, 𝑗::CuDeviceVector{S}, 𝑛::CuDeviceVector{S}, depth) where {T,S}
    
    nbins = size(h, 2)

    it = threadIdx().x
    id, jd = blockDim().x, blockDim().y
    ib, j, k = blockIdx().x, blockIdx().y, blockIdx().z
    ig, jg = gridDim().x, gridDim().y
    
    shared = @cuDynamicSharedMem(T, (size(h, 2), size(h, 4)))
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= length(𝑖) && j <= length(𝑗)
            # depends on shared to be assigned to a single feature
            @inbounds i_idx = 𝑖[i]
            @inbounds n = 𝑛[i_idx]
            # if n != 0
            @inbounds CUDA.atomic_add!(pointer(shared, xid[i_idx, 𝑗[j]] + nbins * (n - 1)), δ[i_idx, k])
            # @inbounds shared[xid[i_idx, 𝑗[j]] + nbins * (n-1)] += δ[i_idx, k]
            # end
        end
        iter += 1
    end
    sync_threads()
    # loop over nodes of given depth
    for nid ∈ 2^(depth - 1):2^(depth) - 1
        # n = 1 # need to loop over nodes
        if it <= nbins
            @inbounds hid = Base._to_linear_index(h, k, it, 𝑗[j], nid)
            @inbounds CUDA.atomic_add!(pointer(h, hid), shared[it, nid])
        end
    end
    sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(
    h::CuArray{T,4}, 
    δ::CuMatrix{T}, 
    X_bin::CuMatrix{UInt8}, 
    𝑖::CuVector{S}, 
    𝑗::CuVector{S}, 
    𝑛::CuVector{S}, depth; 
    MAX_THREADS=256) where {T,S}
    
    fill!(h, 0.0)
    thread_i = min(MAX_THREADS, length(𝑖))
    threads = (thread_i,)
    blocks = (1, length(𝑗), 3)
    # blocks = (ceil(Int, length(𝑖) / thread_i), length(𝑗))
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) * size(h, 4) hist_kernel!(h, δ, X_bin, 𝑖, 𝑗, 𝑛, depth)
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

function update_set_gpu!(𝑛, 𝑖, X_bin, feats, bins, nbins; MAX_THREADS=1024)
    thread_i = min(MAX_THREADS, length(𝑖))
    threads = thread_i
    blocks = length(𝑖) ÷ thread_i + 1
    @cuda blocks = blocks threads = threads update_set_kernel!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    return
end

# split row ids into left and right based on best split condition
function update_set!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    
    @inbounds for i in 𝑖
        feat = feats[𝑛[i]]
        cond = bins[𝑛[i]]
        if cond == nbins
            𝑛[𝑖] = 0
        elseif X_bin[i, feat] <= cond
            𝑛[i] = 𝑛[i] << 1 
        else
            𝑛[i] = 𝑛[i] << 1 + 1
        end
    end
    return nothing
end


# operate on hist_gpu
"""
find_split_gpu!
    Find best split over gpu histograms
"""

function update_gains_gpu!(
    gains::AbstractArray{T,3},
    hist::AbstractArray{T,4}, 
    histL::AbstractArray{T,4}, 
    histR::AbstractArray{T,4},
    𝑗::AbstractVector{S},
    params::EvoTypes,
    nid, depth) where {T,S}

    cumsum!(view(histL, :, :, :, nid), view(hist, :, :, :, nid), dims=2)
    # cumsum!(view(histR, :, :, :, nid), reverse!(view(hist, :, :, :, nid), dims=2), dims=2)
    view(histR, :, :, :, nid) .= view(histL, :, params.nbins:params.nbins, :, nid) .- view(histL, :, :, :, nid)
    hist_gains_gpu!(gains, histL, histR, 𝑗, params.nbins, depth, params.λ)

    return nothing
end


function hist_gains_gpu_kernel!(gains::CuDeviceArray{T,3}, hL::CuDeviceArray{T,4}, hR::CuDeviceArray{T,4}, 𝑗::CuDeviceVector{S}, nbins, depth, λ::T) where {T,S}
    
    i = threadIdx().x
    j = 𝑗[blockIdx().x]
    n = blockIdx().y + 2^(depth - 1) - 1

    @inbounds if hL[3, i, j, n] > 1e-5 && hR[3, i, j, n] > 1e-5
        gains[i, j, n] = (hL[1, i, j, n]^2 / (hL[2, i, j, n] + λ * hL[3, i, j, n]) + 
            hR[1, i, j, n]^2 / (hR[2, i, j, n] + λ * hR[3, i, j, n])) / 2
    elseif i == nbins
        gains[i, j, n] = hL[1, i, j, n]^2 / (hL[2, i, j, n] + λ * hL[3, i, j, n]) / 2 
    end
    return nothing
end

function hist_gains_gpu!(gains::CuArray{T,3}, hL::CuArray{T,4}, hR::CuArray{T,4}, 𝑗::CuVector{S}, nbins, depth, λ::T; MAX_THREADS=256) where {T,S}
    thread_i = min(nbins, MAX_THREADS)
    threads = thread_i
    blocks = length(𝑗), 2^(depth - 1)
    @cuda blocks = blocks threads = threads hist_gains_gpu_kernel!(gains, hL, hR, 𝑗, nbins, depth, λ)
    return gains
end
