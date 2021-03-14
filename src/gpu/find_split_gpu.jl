"""
    build a single histogram containing all grads and weight information
"""
function hist_kernel!(h::CuDeviceArray{T,3}, δ::CuDeviceMatrix{T}, xid::CuDeviceMatrix{S}, 𝑖, 𝑗) where {T,S}
    
    nbins = size(h, 2)

    it = threadIdx().x
    id, jd = blockDim().x, blockDim().y
    ib, j, k = blockIdx().x, blockIdx().y, blockIdx().z
    ig, jg = gridDim().x, gridDim().y
    # j = 1 + (jb - 1) * jd
    
    shared = @cuDynamicSharedMem(T, nbins)
    fill!(shared, 0)
    sync_threads()

    i_tot = length(𝑖)
    iter = 0
    while iter * id * ig < i_tot
        i = it + id * (ib - 1) + iter * id * ig
        if i <= length(𝑖) && j <= length(𝑗)
            # depends on shared to be assigned to a single feature
            @inbounds i_idx = 𝑖[i]
            @inbounds CUDA.atomic_add!(pointer(shared, xid[i_idx, 𝑗[j]]), δ[i_idx, k])   
        end
        iter += 1
    end
    sync_threads()
    # loop to cover cases where nbins > nthreads
    for iter in 1:(nbins - 1) ÷ id + 1
        bin_id = it + id * (iter - 1)
        # bin_id = it
        if bin_id <= nbins
            @inbounds δid = Base._to_linear_index(h, k, bin_id, 𝑗[j])
            @inbounds CUDA.atomic_add!(pointer(h, δid), shared[bin_id])
        end
    end
    # sync_threads()
    return nothing
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(
    h::CuArray{T,3}, 
    δ::CuMatrix{T}, 
    X_bin::CuMatrix{UInt8}, 
    𝑖::CuVector{S}, 
    𝑗::CuVector{S}, 
    K; MAX_THREADS=128) where {T,S}
    
    fill!(h, 0.0)
    thread_i = min(MAX_THREADS, length(𝑖))
    # thread_j = 1
    threads = (thread_i,)
    blocks = (1, length(𝑗), 2 * K + 1)

    @cuda blocks = blocks threads = threads shmem = sizeof(T) * size(h, 2) hist_kernel!(h, δ, X_bin, 𝑖, 𝑗)
    return
end

# update the vector of length 𝑖 pointing to associated node id
function update_set_kernel!(mask, set, best, x_bin)
    it = threadIdx().x
    ibd = blockDim().x
    ibi = blockIdx().x
    i = it + ibd * (ibi - 1)
    @inbounds if i <= length(set)
        @inbounds mask[i] = x_bin[set[i]] <= best
    end
    return nothing
end

function update_set_gpu(set, best, x_bin; MAX_THREADS=1024)
    mask = CUDA.zeros(Bool, length(set))
    thread_i = min(MAX_THREADS, length(set))
    threads = (thread_i,)
    blocks = (length(set) ÷ thread_i + 1,)
    @cuda blocks = blocks threads = threads update_set_kernel!(mask, set, best, x_bin)
    left, right = set[mask], set[.!mask]
    return left, right
end


"""
find_split_gpu! : V1
    Direct translation of the cpu approach.
"""
# function find_split_gpu!(hist_δ::AbstractMatrix{T}, hist_δ²::AbstractMatrix{T}, hist_𝑤::AbstractVector{T},
#     params::EvoTypes, node::TrainNodeGPU{T,S}, info::SplitInfoGPU{T,S}, edges::Vector{T}) where {T,S}

#     # initialize tracking
#     ∑δL = copy(node.∑δ) .* 0
#     ∑δ²L = copy(node.∑δ²) .* 0
#     ∑𝑤L = node.∑𝑤 * 0
#     ∑δR = copy(node.∑δ)
#     ∑δ²R = copy(node.∑δ²)
#     ∑𝑤R = node.∑𝑤

#     # println("∑δ²L: ", ∑δ²L, " ∑δ²R:", ∑δ²R)
#     # println("find_split_gpu! hist_𝑤: ", hist_𝑤)
#     # println("∑𝑤L: ", ∑𝑤L, " ∑𝑤R: ", ∑𝑤R)

#     @inbounds for bin in 1:(length(hist_δ) - 1)
#         @views ∑δL .+= hist_δ[:, bin]
#         @views ∑δ²L .+= hist_δ²[:, bin]
#         ∑𝑤L += hist_𝑤[bin]
#         @views ∑δR .-= hist_δ[:, bin]
#         @views ∑δ²R .-= hist_δ²[:, bin]
#         ∑𝑤R -= hist_𝑤[bin]

#         # println("∑δ²L: ", ∑δ²L, " | ∑δ²R:", ∑δ²R, " | hist_δ²[bin,:]: ", hist_δ²[bin,:])

#         gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
#         gain = gainL + gainR

#         # println("∑𝑤L: ", ∑𝑤L, " ∑𝑤R: ", ∑𝑤R)
#         # println("∑δL: ", ∑δL, " ∑δR: ", ∑δR)
#         # println("info.gain: ", info.gain, " gain: ", gain)

#         if gain > info.gain && ∑𝑤L >= params.min_weight + 0.1 && ∑𝑤R >= params.min_weight + 0.1
#             # println("there's a gain on bin: ", bin)
#             info.gain = gain
#             info.gainL = gainL
#             info.gainR = gainR
#             @views info.∑δL .= ∑δL
#             @views info.∑δ²L .= ∑δ²L
#             info.∑𝑤L = ∑𝑤L
#             @views info.∑δR .= ∑δR
#             @views info.∑δ²R .= ∑δ²R
#             info.∑𝑤R = ∑𝑤R
#             info.cond = edges[bin]
#             info.𝑖 = bin
#         end # info update if gain
#     end # loop on bins
# end


# operate on hist_gpu
"""
find_split_gpu!
    Find best split over gpu histograms
    ! Check for behavior when sme histograms are empty / near zero observations
"""
function find_split_gpu!(hist::AbstractArray{T,3}, edges::Vector{Vector{T}}, params::EvoTypes) where {T}

    hist_cum_L = cumsum(hist, dims=2)
    hist_cum_R = sum(hist, dims=2) .- hist_cum_L
    
    gains_L = get_hist_gains_gpu(hist_cum_L, params.λ)
    gains_R = get_hist_gains_gpu(hist_cum_R, params.λ)
    gains = gains_L + gains_R

    best = findmax(gains)
    gain, bin, feat = best[1], best[2][1], UInt32(best[2][2])
    cond = edges[feat][bin]
    gainL, gainR = gains_L[bin, feat], gains_R[bin, feat]

    ∑L = Array(hist_cum_L[:, bin, feat])
    ∑R = Array(hist_cum_R[:, bin, feat])

    return (gain = gain, bin = bin, feat = feat, cond = cond,
        gainL = gainL, gainR = gainR,
        ∑L = ∑L, ∑R = ∑R)
end


function hist_gains_gpu!(gains::CuDeviceMatrix{T}, h::CuDeviceArray{T,3}, λ::T) where {T}
    
    i, j = threadIdx().x, blockIdx().y
    K = (size(h, 1) - 1) ÷ 2

    @inbounds 𝑤 = h[2 * K + 1, i, j] 
    
    @inbounds if 𝑤 > 1e-8
        @inbounds for k in 1:K
            @inbounds gains[i, j] += (h[k, i, j]^2 / (h[2 * K + k - 1, i, j] + λ * 𝑤)) / 2
        end
    end

    return nothing
end

function get_hist_gains_gpu(h::CuArray{T,3}, λ::T; MAX_THREADS=1024) where {T}
    
    gains = CUDA.zeros(T, size(h, 1) - 1, size(h, 2))

    thread_i = min(size(gains, 1), MAX_THREADS)
    thread_j = 1
    threads = (thread_i, thread_j)
    blocks = (1, size(gains, 2))

    @cuda blocks = blocks threads = threads hist_gains_gpu!(gains, h, λ)
    return gains
end
