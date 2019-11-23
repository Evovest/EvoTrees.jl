using CuArrays

features = rand(1_000, 10)
features_int = rand(UInt8, 1_000, 10)

hist = rand(256)
δ = rand(1_000_000)
idx = rand(UInt8, 1_000_000)
set = collect(1:length(vec))

hist_gpu = cu(hist)
δ_gpu = cu(δ)
idx_gpu = cu(idx)
set_gpu = cu(set)

function split_cpu(hist, δ, idx, set)
    @inbounds for i in set
        hist[idx[i]] += δ[i]
    end
    return
end

function split_gpu(hist, δ, idx, set)
    @inbounds for i in set
        hist[idx[i]] += δ[i]
    end
    return
end


hist
@time split_cpu(hist, δ, idx, set)
@time split_gpu(hist, δ_gpu, idx, set)
