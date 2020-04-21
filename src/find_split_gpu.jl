# GPU - apply along the features axis
function kernel_v1!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    @inbounds if i <= size(id, 1) && j <= size(h, 2)
        k = Base._to_linear_index(h, id[i,j], j)
        CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

function hist_gpu_v1!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}; MAX_THREADS=512) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(h, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(h, 2)) ./ threads)
    CuArrays.@cuda blocks=blocks threads=threads kernel_v1!(h, x, id)
    return h
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
    # @inbounds @threads for j in node.𝑗
    #     @inbounds for i in node.𝑖
    #         hist_δ[X_bin[i,j], j] += δ[i]
    #         hist_δ²[X_bin[i,j], j] += δ²[i]
    #         hist_𝑤[X_bin[i,j], j] += 𝑤[i]
    #     end
    # end
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


function find_split_gpu_test!(hist_δ::Vector{SVector{L,T}}, hist_δ²::Vector{SVector{L,T}}, hist_𝑤::Vector{SVector{1,T}}, bins::Vector{BitSet}, X_bin, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, params::EvoTreeRegressor, info::SplitInfo{L,T,S}, edges::Vector{T}, set::Vector{S}) where {L,T,S}

    # initialize histogram
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    # build histogram
    @inbounds for i in set
        hist_δ[X_bin[i]] += δ[i]
        hist_δ²[X_bin[i]] += δ²[i]
        hist_𝑤[X_bin[i]] += 𝑤[i]
    end
    return
end
