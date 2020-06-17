# GPU - apply along the features axis
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id, 𝑖, 𝑗) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= length(𝑖) && j <= length(𝑗)
        @inbounds k = Base._to_linear_index(h, id[𝑖[i], 𝑗[j]], 𝑗[j])
        @inbounds CUDAnative.atomic_add!(pointer(h, k), x[𝑖[i], 𝑗[j]])
    end
    return
end

# base approach - block built along the cols first, the rows (limit collisions)
function hist_gpu!(h::CuMatrix{T}, x::CuMatrix{T}, id, 𝑖, 𝑗; MAX_THREADS=1024) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, length(𝑗))
    thread_i = min(MAX_THREADS ÷ thread_j, length(𝑖))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (length(𝑖), length(𝑗)) ./ threads)
    CuArrays.@sync begin
        @cuda blocks=blocks threads=threads kernel!(h, x, id, 𝑖, 𝑗)
    end
    return
end

function update_hist_gpu!(hist_δ::Matrix{SVector{L,T}}, hist_δ²::Matrix{SVector{L,T}}, hist_𝑤::Matrix{SVector{1,T}},
    δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}},
    X_bin, node::TrainNode{L,T,S}) where {L,T,S}

    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    hist_gpu_v1!(hist_δ, δ, id)
    hist_gpu_v1!(hist_δ², δ², id)
    hist_gpu_v1!(hist_𝑤, 𝑤, id)
end

function find_split_gpu!(hist_δ::AbstractVector{SVector{L,T}}, hist_δ²::AbstractVector{SVector{L,T}}, hist_𝑤::AbstractVector{SVector{1,T}},
    params::EvoTypes, node::TrainNode{L,T,S}, info::SplitInfo{L,T,S}, edges::Vector{T}) where {L,T,S}

    # initialize tracking
    ∑δL = node.∑δ * 0
    ∑δ²L = node.∑δ² * 0
    ∑𝑤L = node.∑𝑤 * 0
    ∑δR = node.∑δ
    ∑δ²R = node.∑δ²
    ∑𝑤R = node.∑𝑤

    @inbounds for bin in 1:(length(hist_δ)-1)
        ∑δL += hist_δ[bin]
        ∑δ²L += hist_δ²[bin]
        ∑𝑤L += hist_𝑤[bin]
        ∑δR -= hist_δ[bin]
        ∑δ²R -= hist_δ²[bin]
        ∑𝑤R -= hist_𝑤[bin]

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        if gain > info.gain && ∑𝑤L[1] >= params.min_weight + 1e-12 && ∑𝑤R[1] >= params.min_weight + 1e-12
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            info.∑δL = ∑δL
            info.∑δ²L = ∑δ²L
            info.∑𝑤L = ∑𝑤L
            info.∑δR = ∑δR
            info.∑δ²R = ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end # info update if gain
    end # loop on bins
end
