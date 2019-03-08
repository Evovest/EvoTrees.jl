using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using BenchmarkTools
using Profile
using StatsBase: sample

using Revise
using Traceur
using EvoTrees
using EvoTrees: get_gain, update_gains!, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, Params, predict, predict!, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Array, features)
Y = data[54]
Y = convert(Array{Float64}, Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# idx
X_perm = zeros(Int, size(X))
@threads for feat in 1:size(X, 2)
    X_perm[:, feat] = sortperm(X[:, feat]) # returns gain value and idx split
    # idx[:, feat] = sortperm(view(X, :, feat)) # returns gain value and idx split
end

# placeholder for sort perm
perm_ini = zeros(Int, size(X))


params1 = Params(:linear, 1, 1.0, 0.1, 1.0, 5, 5.0, 0.8, 0.9)
Val{params1.loss}()
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
pred = zeros(size(Y, 1))
@time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
∑δ, ∑δ² = sum(δ), sum(δ²)
gain = get_gain(∑δ, ∑δ², params1.λ)

splits = Vector{SplitInfo}(undef, size(X, 2))
for feat in 1:size(X, 2)
    splits[feat] = SplitInfo(-Inf, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
end
tracks = Vector{SplitTrack}(undef, size(X, 2))
for feat in 1:size(X, 2)
    tracks[feat] = SplitTrack(0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
end

x = X[:, 5]
x_sortperm = sortperm(x)
x_sort = x[x_sortperm]
δ_sort = δ[x_sortperm]
δ²_sort = δ²[x_sortperm]


function find_split_1(x::AbstractArray, δ::AbstractArray, δ²::AbstractArray, ∑δ, ∑δ², λ, info::SplitInfo, track::SplitTrack)

    info.gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²

    @inbounds for i in 1:(size(x, 1) - 1)

        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]

        @inbounds if x[i] < x[i+1] # check gain only if there's a change in value
            update_track!(track, λ)
            if track.gain > info.gain
                info.gain = track.gain
                info.gainL = track.gainL
                info.gainR = track.gainR
                info.∑δL = track.∑δL
                info.∑δ²L = track.∑δ²L
                info.∑δR = track.∑δR
                info.∑δ²R = track.∑δ²R
                info.cond = x[i]
                info.𝑖 = i
            end
        end
    end
end


@time split_1 = find_split_1(x_sort, δ_sort, δ²_sort, ∑δ, ∑δ², params1.λ, splits[1], tracks[1])
@code_warntype find_split_1(x_sort, δ_sort, δ²_sort, ∑δ, ∑δ², params1.λ, splits[1], tracks[1])
splits[1]
