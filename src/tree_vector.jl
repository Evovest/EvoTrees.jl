# initialize train_nodes
function grow_tree(X::AbstractArray{T, 2}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, 𝑤::AbstractArray{Float64, 1}, params::Params, perm_ini::AbstractArray{Int}, train_nodes::Vector{TrainNode}, splits::Vector{SplitInfo{Float64}}, tracks::Vector{SplitTrack{Float64}}) where {T<:Real}

    active_id = ones(Int, 1)
    leaf_count = 1::Int
    tree_depth = 1::Int

    tree = Tree(Vector{TreeNode{Float64, Int}}())

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)

        # grow nodes
        for id in active_id

            node = train_nodes[id]

            if tree_depth == params.max_depth
                push!(tree.nodes, TreeNode(- params.η * node.∑δ / (node.∑δ² + params.λ * node.∑𝑤)))
            else
                node_size = size(node.𝑖, 1)
                @threads for feat in node.𝑗
                # for feat in node.𝑗
                    sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = false)
                    find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params.λ, splits[feat], tracks[feat])
                    splits[feat].feat = feat
                end

                # assign best split
                best = get_max_gain(splits)

                # grow node if best split improve gain
                if best.gain > node.gain + params.γ
                    # Node: depth, ∑δ, ∑δ², gain, 𝑖, 𝑗
                    train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.∑𝑤L, best.gainL, node.𝑖[perm_ini[1:best.𝑖, best.feat]], node.𝑗[:])
                    train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.∑𝑤R, best.gainR, node.𝑖[perm_ini[best.𝑖+1:node_size, best.feat]], node.𝑗[:])
                    # push split Node
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond))
                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)
                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode(- params.η * node.∑δ / (node.∑δ² + params.λ * node.∑𝑤)))
                end # end of single node split search
            end
            # node.𝑖 = [0]
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits::Vector{SplitInfo{Float64}})
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    # best.feat = feat
    return best
end

# grow_gbtree
function grow_gbtree(X::AbstractArray{T, 2}, Y::AbstractArray{<:AbstractFloat, 1}, params::Params; X_eval::AbstractArray{T, 2} = Array{T, 2}(undef, (0,0)), Y_eval::AbstractArray{<:AbstractFloat, 1} = Array{Float64, 1}(undef, 0), metric::Symbol = :rmse)  where T<:Real
    μ = mean(Y)
    # pred = ones(size(Y, 1)) .* μ
    pred = ones(size(Y, 1)) .* μ

    # initialize gradients and weights
    δ, δ² = zeros(Float64, size(Y, 1)), zeros(Float64, size(Y, 1))
    𝑤 = ones(Float64, size(Y, 1))
    update_grads!(Val{params.loss}(), pred, Y, δ, δ², 𝑤)

    # eval init
    if size(Y_eval, 1) > 0
        # pred_eval = ones(size(Y_eval, 1)) .* μ
        pred_eval = ones(size(Y_eval, 1)) .* μ
    end

    bias = Tree([TreeNode(μ)])
    gbtree = GBTree([bias], params)

    # sort perm id placeholder
    perm_ini = zeros(Int, size(X))

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    train_nodes = Vector{TrainNode}(undef, 2^params.max_depth-1)
    for feat in 1:2^params.max_depth-1
        train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, [0], [0])
    end

    # loop over nrounds
    for i in 1:params.nrounds
        # select random rows and cols
        𝑖 = 𝑖_[sample(𝑖_, floor(Int, params.rowsample * X_size[1]), replace = false)]
        𝑗 = 𝑗_[sample(𝑗_, floor(Int, params.colsample * X_size[2]), replace = false)]

        # get gradients
        update_grads!(Val{params.loss}(), pred, Y, δ, δ², 𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
        gain = get_gain(∑δ, ∑δ², ∑𝑤, params.λ)

        # initializde node splits info and tracks - colsample size (𝑗)
        splits = Vector{SplitInfo{Float64}}(undef, X_size[2])
        for feat in 𝑗_
            splits[feat] = SplitInfo{Float64}(-Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
        end
        tracks = Vector{SplitTrack{Float64}}(undef, X_size[2])
        for feat in 𝑗_
            tracks[feat] = SplitTrack{Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
        end

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(X, δ, δ², 𝑤, params, perm_ini, train_nodes, splits, tracks)
        # update push tree to model
        push!(gbtree.trees, tree)

        # get update predictions
        predict!(pred, tree, X)
        # eval predictions
        if size(Y_eval, 1) > 0
            predict!(pred_eval, tree, X_eval)
        end

        # callback function
        if mod(i, 10) == 0
            if size(Y_eval, 1) > 0
                display(string("iter:", i, ", train:", eval_metric(Val{metric}(), pred, Y), ", eval: ", eval_metric(Val{metric}(), pred_eval, Y_eval)))
                # println("iter:", i, ", train:", eval_metric(Val{metric}(), pred, Y), ", eval: ", eval_metric(Val{metric}(), pred_eval, Y_eval))
            else
                display(string("iter:", i, ", train:", eval_metric(Val{metric}(), pred, Y)))
                # println("iter:", i, ", train:", eval_metric(Val{metric}(), pred, Y))
            end
        end # end of callback

    end #end of nrounds
    return gbtree
end

# find best split
function find_split!(x::AbstractArray{T, 1}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, 𝑤::AbstractArray{Float64, 1}, ∑δ, ∑δ², ∑𝑤, λ, info::SplitInfo, track::SplitTrack) where T<:Real

    # info.gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0
    @fastmath info.gain = (∑δ ^ 2 / (∑δ² + λ .* ∑𝑤)) / 2.0

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑𝑤L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    @fastmath @inbounds for i in 1:(size(x, 1) - 1)
    # @fastmath @inbounds for i in eachindex(x)

        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑𝑤L += 𝑤[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]
        track.∑𝑤R -= 𝑤[i]

        @fastmath @inbounds if x[i] < x[i+1] # check gain only if there's a change in value

            update_track!(track, λ)
            if track.gain > info.gain
                info.gain = track.gain
                info.gainL = track.gainL
                info.gainR = track.gainR
                info.∑δL = track.∑δL
                info.∑δ²L = track.∑δ²L
                info.∑𝑤L = track.∑𝑤L
                info.∑δR = track.∑δR
                info.∑δ²R = track.∑δ²R
                info.∑𝑤R = track.∑𝑤R
                info.cond = x[i]
                info.𝑖 = i
            end
        end
    end
end
