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
    blocks = (16, length(𝑗))
    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 1) * size(h, 2) hist_kernel!(h, δ𝑤, X_bin, 𝑖, 𝑗)
    CUDA.synchronize()
    return
end

"""
    Multi-threads split_set!
        Take a view into left and right placeholders. Right ids are assigned at the end of the length of the current node set.
"""
function split_chunk_kernel!(left::CuDeviceVector{S}, right::CuDeviceVector{S}, 𝑖::CuDeviceVector{S}, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights) where {S}

    it = threadIdx().x
    bid = blockIdx().x
    gdim = gridDim().x

    left_count = 0
    right_count = 0

    i_size = length(𝑖)
    i = it + chunk_size * (bid - 1)
    
    bid == gdim ? bsize = i_size - chunk_size * (bid - 1) : bsize = chunk_size
    i_max = i + bsize - 1

    while i <= i_max
        if X_bin[𝑖[i], feat] <= cond_bin
            left_count += 1
            left[offset + chunk_size * (bid - 1) + left_count] = 𝑖[i]
        else
            right_count += 1
            right[offset + chunk_size * (bid - 1) + right_count] = 𝑖[i]
        end
        i += 1
    end

    lefts[bid] = left_count
    rights[bid] = right_count

    return nothing
end

function split_views_kernel!(out::CuDeviceVector{S}, left::CuDeviceVector{S}, right::CuDeviceVector{S}, offset, lefts, rights) where {S}    

    it = threadIdx().x
    
    sum_lefts = sum(lefts)
    left_cum = 0
    right_cum = 0

    @inbounds for bid in eachindex(lefts)
        view(out, offset + left_cum + 1:offset + left_cum + lefts[bid]) .= view(left, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + lefts[bid])
        view(out, offset + sum_lefts + right_cum + 1:offset + sum_lefts + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        left_cum += lefts[bid]
        right_cum += rights[bid]
    end

    return (view(out, offset + 1:offset + sum_lefts), view(out, offset + sum_lefts + 1:offset + length(𝑖)))
end

function split_set_threads_gpu!(out, left, right, 𝑖, X_bin, feat, cond_bin, offset)
    𝑖_size = length(𝑖)
    
    chunk_size = min(𝑖_size, 1024)
    nblocks = ceil(Int, 𝑖_size / chunk_size)

    lefts = CUDA.zeros(Int, nblocks)
    rights = CUDA.zeros(Int, nblocks)

    threads = 1

    @cuda blocks = nblocks threads = threads split_chunk_kernel!(left, right, 𝑖, X_bin, feat, cond_bin, offset, chunk_size, lefts, rights)
    CUDA.synchronize()
    
    # @cuda blocks = 1 threads = 1 split_views_kernel!(out, left, right, offset, lefts, rights)
    # CUDA.synchronize()    
    sum_lefts = sum(lefts)
    left_cum = 0
    right_cum = 0
    @inbounds for bid in eachindex(lefts)
        view(out, offset + left_cum + 1:offset + left_cum + lefts[bid]) .= view(left, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + lefts[bid])
        view(out, offset + sum_lefts + right_cum + 1:offset + sum_lefts + right_cum + rights[bid]) .= view(right, offset + chunk_size * (bid - 1) + 1:offset + chunk_size * (bid - 1) + rights[bid])
        left_cum += lefts[bid]
        right_cum += rights[bid]
    end
    
    return (view(out, offset + 1:offset + sum_lefts), view(out, offset + sum_lefts + 1:offset + length(𝑖)))
    
    return nothing
end


"""
    update_gains!
        Generic fallback
"""
function update_gains_gpu!(
    loss::L,
    node::TrainNodeGPU{T},
    𝑗::AbstractVector{S},
    params::EvoTypes, K;
    MAX_THREADS=512) where {L,T,S}

    cumsum!(view(node.hL, :, :, :), view(node.h, :, :, :), dims=2)
    # cumsum!(view(histR, :, :, :, nid), reverse!(view(hist, :, :, :, nid), dims=2), dims=2)
    view(node.hR, :, :, :) .= view(node.hL, :, params.nbins:params.nbins, :) .- view(node.hL, :, :, :)

    thread_i = min(params.nbins, MAX_THREADS)
    threads = thread_i
    blocks = length(𝑗)
    @cuda blocks = blocks threads = threads hist_gains_gpu_kernel!(node.gains, node.hL, node.hR, 𝑗, params.nbins, params.λ)
    # hist_gains_gpu!(loss, node.gains, node.hL, node.hR, 𝑗, params.nbins, params.λ)
    CUDA.synchronize()
    return nothing
end

function hist_gains_gpu_kernel!(gains::CuDeviceMatrix{T}, hL::CuDeviceArray{T,3}, hR::CuDeviceArray{T,3}, 𝑗::CuDeviceVector{S}, nbins, λ::T) where {T,S}

    i = threadIdx().x
    @inbounds j = 𝑗[blockIdx().x]

    @inbounds if hL[3, i, j] > 1e-5 && hR[3, i, j] > 1e-5
        gains[i, j] = (hL[1, i, j]^2 / (hL[2, i, j] + λ * hL[3, i, j]) + 
        hR[1, i, j]^2 / (hR[2, i, j,] + λ * hR[3, i, j])) / 2
        elseif i == nbins
            gains[i, j] = hL[1, i, j]^2 / (hL[2, i, j] + λ * hL[3, i, j]) / 2 
        end
    return nothing
end

# function hist_gains_gpu!(loss::L, gains::CuMatrix{T}, hL::CuArray{T,3}, hR::CuArray{T,3}, 𝑗::CuVector{S}, nbins, λ::T; MAX_THREADS=512) where {L <: GradientRegression,T,S}
#     thread_i = min(nbins, MAX_THREADS)
#     threads = thread_i
#     blocks = length(𝑗)
#     @cuda blocks = blocks threads = threads hist_gains_gpu_kernel!(gains, hL, hR, 𝑗, nbins, λ)
#     return gains
# end
