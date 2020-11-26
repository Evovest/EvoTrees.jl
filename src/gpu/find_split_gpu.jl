# # GPU - apply along the features axis
# function hist_kernel!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗, K) where {T <: AbstractFloat}
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
#     if i <= length(𝑖) && j <= length(𝑗)
#         for k in 1:K
#             @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], k, 𝑗[j])
#             @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i],k])
#         end
#     end
#     return
# end

# # for 2D input: 𝑤
# function hist_kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T <: AbstractFloat}
#     i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
#     if i <= length(𝑖) && j <= length(𝑗)
#         @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
#         @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i]])
#     end
#     return
# end

# base approach - block built along the cols first, the rows (limit collisions)
# function update_hist_gpu!(hδ::CuArray{T,3}, hδ²::CuArray{T,3}, h𝑤::CuMatrix{T},
#     δ::CuMatrix{T}, δ²::CuMatrix{T}, 𝑤::CuVector{T},
#     X_bin::CuMatrix{Int}, 𝑖::CuVector{Int}, 𝑗::CuVector{Int}, K; MAX_THREADS=1024) where {T <: AbstractFloat}

#     hδ .= T(0.0)
#     hδ² .= T(0.0)
#     h𝑤 .= T(0.0)

#     thread_j = min(MAX_THREADS, length(𝑗))
#     thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
#     threads = (thread_i, thread_j)
#     blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
#     @cuda blocks = blocks threads = threads hist_kernel!(hδ, δ, X_bin, 𝑖, 𝑗, K)
#     @cuda blocks = blocks threads = threads hist_kernel!(hδ², δ², X_bin, 𝑖, 𝑗, K)
#     @cuda blocks = blocks threads = threads hist_kernel!(h𝑤, 𝑤, X_bin, 𝑖, 𝑗)

#     return
# end

function hist_kernel!(hδ1::CuDeviceArray{T,3}, hδ2::CuDeviceArray{T,3}, h𝑤::CuDeviceMatrix{T}, δ1::CuDeviceMatrix{T}, δ2::CuDeviceMatrix{T}, 𝑤::CuDeviceVector{T}, xid::CuDeviceMatrix{S}, 𝑖, 𝑗) where {T,S}
    
    nbins = size(h𝑤, 1)
    it, jt = threadIdx().x, threadIdx().y
    ib, jb = blockIdx().x, blockIdx().y
    id, jd = blockDim().x, blockDim().y
    ig, jg = gridDim().x, gridDim().y
    j = jt + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, 3 * nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= length(𝑖) && j <= length(𝑗)
            # depends on shared to be assigned to a single feature
            i_idx = 𝑖[i]
            k = 3 * (xid[i_idx, 𝑗[j]] - 1)
            @inbounds CUDA.atomic_add!(pointer(shared, k + 1), δ1[i_idx, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 2), δ2[i_idx, 1])
            @inbounds CUDA.atomic_add!(pointer(shared, k + 3), 𝑤[i_idx])
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for iter in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (iter - 1)
        if bin_id <= nbins
            @inbounds k = Base._to_linear_index(hδ1, 1, bin_id, 𝑗[j])
            @inbounds CUDA.atomic_add!(pointer(hδ1, k), shared[3 * (bin_id - 1) + 1])
            @inbounds CUDA.atomic_add!(pointer(hδ2, k), shared[3 * (bin_id - 1) + 2])
            @inbounds CUDA.atomic_add!(pointer(h𝑤, k), shared[3 * (bin_id - 1) + 3])
        end
    end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(hδ::CuArray{T,3}, hδ²::CuArray{T,3}, h𝑤::CuMatrix{T},
        δ::CuMatrix{T}, δ²::CuMatrix{T}, 𝑤::CuVector{T},
        X_bin::CuMatrix{UInt8}, 𝑖::CuVector{S}, 𝑗::CuVector{S}, K; MAX_THREADS=128) where {T,S}
    
    fill!(hδ, 0.0)
    fill!(hδ², 0.0)
    fill!(h𝑤, 0.0)

    thread_i = min(MAX_THREADS, length(𝑖))
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = (8, length(𝑗))

    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h𝑤, 1) * 3 hist_kernel!(hδ, hδ², h𝑤, δ, δ², 𝑤, X_bin, 𝑖, 𝑗)
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
# function update_hist_gpu!(hδ::CuArray{T,3}, hδ²::CuArray{T,3}, h𝑤::CuMatrix{T},
#     δ::CuMatrix{T}, δ²::CuMatrix{T}, 𝑤::CuVector{T},
#     X_bin::CuMatrix{Int}, 𝑖::CuVector{Int}, 𝑗::CuVector{Int}, K; MAX_THREADS=1024) where {T <: AbstractFloat}

#     hδ .= T(0.0)
#     hδ² .= T(0.0)
#     h𝑤 .= T(0.0)

#     thread_j = min(MAX_THREADS, length(𝑗))
#     thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
#     threads = (thread_i, thread_j)
#     blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
#     @cuda blocks = blocks threads = threads hist_kernel!(hδ, δ, X_bin, 𝑖, 𝑗, K)
#     @cuda blocks = blocks threads = threads hist_kernel!(hδ², δ², X_bin, 𝑖, 𝑗, K)
#     @cuda blocks = blocks threads = threads hist_kernel!(h𝑤, 𝑤, X_bin, 𝑖, 𝑗)

#     return
# end

function find_split_gpu!(hist_δ::AbstractMatrix{T}, hist_δ²::AbstractMatrix{T}, hist_𝑤::AbstractVector{T},
    params::EvoTypes, node::TrainNode_gpu{T,S}, info::SplitInfo_gpu{T,S}, edges::Vector{T}) where {T,S}

    # initialize tracking
    ∑δL = copy(node.∑δ) .* 0
    ∑δ²L = copy(node.∑δ²) .* 0
    ∑𝑤L = node.∑𝑤 * 0
    ∑δR = copy(node.∑δ)
    ∑δ²R = copy(node.∑δ²)
    ∑𝑤R = node.∑𝑤

    # println("∑δ²L: ", ∑δ²L, " ∑δ²R:", ∑δ²R)
    # println("find_split_gpu! hist_𝑤: ", hist_𝑤)
    # println("∑𝑤L: ", ∑𝑤L, " ∑𝑤R: ", ∑𝑤R)

    @inbounds for bin in 1:(length(hist_δ) - 1)
        @views ∑δL .+= hist_δ[:, bin]
        @views ∑δ²L .+= hist_δ²[:, bin]
        ∑𝑤L += hist_𝑤[bin]
        @views ∑δR .-= hist_δ[:, bin]
        @views ∑δ²R .-= hist_δ²[:, bin]
        ∑𝑤R -= hist_𝑤[bin]

        # println("∑δ²L: ", ∑δ²L, " | ∑δ²R:", ∑δ²R, " | hist_δ²[bin,:]: ", hist_δ²[bin,:])

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        # println("∑𝑤L: ", ∑𝑤L, " ∑𝑤R: ", ∑𝑤R)
        # println("∑δL: ", ∑δL, " ∑δR: ", ∑δR)
        # println("info.gain: ", info.gain, " gain: ", gain)

        if gain > info.gain && ∑𝑤L >= params.min_weight + 0.1 && ∑𝑤R >= params.min_weight + 0.1
            # println("there's a gain on bin: ", bin)
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            @views info.∑δL .= ∑δL
            @views info.∑δ²L .= ∑δ²L
            info.∑𝑤L = ∑𝑤L
            @views info.∑δR .= ∑δR
            @views info.∑δ²R .= ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end # info update if gain
    end # loop on bins
end
