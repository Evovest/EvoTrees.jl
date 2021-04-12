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

# split row ids into left and right based on best split condition
function update_set!(𝑛, 𝑖, X_bin, feats, bins, nbins)
    
    @inbounds for i in 𝑖
        feat = feats[𝑛[i]]
        cond = bins[𝑛[i]]
        if cond == nbins
            𝑛[i] = 0
        elseif X_bin[i, feat] <= cond
            𝑛[i] = 𝑛[i] << 1 
        else
            𝑛[i] = 𝑛[i] << 1 + 1
        end
    end
    return nothing
end


function update_hist!(
    hist::Array{T,4}, 
    δ::Matrix{T}, 
    X_bin::Matrix{UInt8}, 
    𝑖::Vector{S}, 
    𝑗::Vector{S}, 
    𝑛::Vector{S}) where {T,S}
    
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            if 𝑛[i] != 0
                hist[1, X_bin[i, j], j, 𝑛[i]] += δ[i, 1]
                hist[2, X_bin[i, j], j, 𝑛[i]] += δ[i, 2]
                hist[3, X_bin[i, j], j, 𝑛[i]] += δ[i, 3]
            end
        end
    end
end


function update_gains!(
    gains::AbstractArray{T,3},
    hist::AbstractArray{T,4}, 
    histL::AbstractArray{T,4}, 
    histR::AbstractArray{T,4},
    𝑗::Vector{S},
    params::EvoTypes,
    nid) where {T,S}

    cumsum!(view(histL, :, :, :, nid), view(hist, :, :, :, nid), dims=2)
    # cumsum!(view(histR, :, :, :, nid), reverse!(view(hist, :, :, :, nid), dims=2), dims=2)
    view(histR, :, :, :, nid) .= view(histL, :, params.nbins:params.nbins, :, nid) .- view(histL, :, :, :, nid)
    hist_gains_cpu!(gains, histL, histR, 𝑗, params.nbins, nid, params.λ)

    return nothing
end


function hist_gains_cpu!(gains::Array{T,3}, hL::Array{T,4}, hR::Array{T,4}, 𝑗::Vector{S}, nbins, nodes, λ::T) where {T,S}
    @inbounds for n in nodes
        @inbounds for j in 𝑗
            @inbounds for i in 1:nbins
                # update gain only if there's non null weight on each of left and right side - except for nbins level, which is used as benchmark for split criteria (gain if no split)
                if hL[3, i, j, n] > 1e-5 && hR[3, i, j, n] > 1e-5
                    @inbounds gains[i, j, n] = (hL[1, i, j, n]^2 / (hL[2, i, j, n] + λ * hL[3, i, j, n]) + 
                        hR[1, i, j, n]^2 / (hR[2, i, j, n] + λ * hR[3, i, j, n])) / 2
                elseif i == nbins
                    @inbounds gains[i, j, n] = hL[1, i, j, n]^2 / (hL[2, i, j, n] + λ * hL[3, i, j, n]) / 2 
                end
            end
        end
    end
    return nothing
end
