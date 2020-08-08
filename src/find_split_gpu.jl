# GPU - apply along the features axis
function hist_kernel!(h::CuDeviceArray{T,3}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗, K) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        for k in 1:K
            @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], k, 𝑗[j])
            @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i],k])
        end
    end
    return
end

# for 2D input like 𝑤
function hist_kernel!(h::CuDeviceMatrix{T}, x::CuDeviceVector{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds pt = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDA.atomic_add!(pointer(h, pt), x[𝑖[i]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function update_hist_gpu!(hδ::CuArray{T,3}, hδ²::CuArray{T,3}, h𝑤::CuMatrix{T},
    δ::CuMatrix{T}, δ²::CuMatrix{T}, 𝑤::CuVector{T},
    X_bin::CuMatrix{Int}, 𝑖::CuVector{Int}, 𝑗::CuVector{Int}, K; MAX_THREADS=1024) where {T<:AbstractFloat}

    hδ .*= 0.0
    hδ² .*= 0.0
    h𝑤 .*= 0.0

    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    @cuda blocks=blocks threads=threads hist_kernel!(hδ, δ, X_bin, 𝑖, 𝑗, K)
    @cuda blocks=blocks threads=threads hist_kernel!(hδ², δ², X_bin, 𝑖, 𝑗, K)
    @cuda blocks=blocks threads=threads hist_kernel!(h𝑤, 𝑤, X_bin, 𝑖, 𝑗)

    return
end
#
# # base approach - block built along the cols first, the rows (limit collisions)
# function hist_gpu!(h::CuMatrix{T}, x::CuVector{T}, X_bin::CuMatrix{Int}, 𝑖::CuVector{Int}, 𝑗::CuVector{Int}; MAX_THREADS=1024) where {T<:AbstractFloat}
#     thread_j = min(MAX_THREADS, length(𝑗))
#     thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
#     threads = (thread_i, thread_j)
#     blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
#     @cuda blocks=blocks threads=threads hist_kernel!(h, x, X_bin, 𝑖, 𝑗)
#     return
# end
#
# function update_hist_gpu!(hist_δ::AbstractArray{T,3}, hist_δ²::AbstractArray{T,3}, hist_𝑤::AbstractMatrix{T},
#     δ::AbstractMatrix{T}, δ²::AbstractMatrix{T}, 𝑤::AbstractVector{T},
#     X_bin::AbstractMatrix{Int}, node::TrainNode_gpu{T,S}, K) where {T,S}
#
#     hist_δ .*= 0.0
#     hist_δ² .*= 0.0
#     hist_𝑤 .*= 0.0
#
#     hist_gpu!(hist_δ, δ, X_bin, CuArray(node.𝑖), CuArray(node.𝑗), K)
#     hist_gpu!(hist_δ², δ², X_bin, CuArray(node.𝑖), CuArray(node.𝑗), K)
#     hist_gpu!(hist_𝑤, 𝑤, X_bin, CuArray(node.𝑖), CuArray(node.𝑗))
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
    # println("hist_δ²: ", hist_δ²)

    @inbounds for bin in 1:(length(hist_δ)-1)
        ∑δL .+= hist_δ[bin,:]
        ∑δ²L .+= hist_δ²[bin,:]
        ∑𝑤L += hist_𝑤[bin]
        ∑δR .-= hist_δ[bin,:]
        ∑δ²R .-= hist_δ²[bin,:]
        ∑𝑤R -= hist_𝑤[bin]

        # println("∑δ²L: ", ∑δ²L, " | ∑δ²R:", ∑δ²R, " | hist_δ²[bin,:]: ", hist_δ²[bin,:])

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        if gain > info.gain && ∑𝑤L >= params.min_weight + 1e-12 && ∑𝑤R >= params.min_weight + 1e-12
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            info.∑δL .= ∑δL
            info.∑δ²L .= ∑δ²L
            info.∑𝑤L = ∑𝑤L
            info.∑δR .= ∑δR
            info.∑δ²R .= ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end # info update if gain
    end # loop on bins
end
