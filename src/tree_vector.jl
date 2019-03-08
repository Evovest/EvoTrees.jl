# initialize train_nodes
function grow_tree(X::AbstractArray{T, 2}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, params::Params, perm_ini::AbstractArray{Int}, train_nodes::Vector{TrainNode}) where T<:Real

    active_id = [1]
    leaf_count = 1
    tree_depth = 1

    tree = Tree(Vector{TreeNode}())

    splits = Vector{SplitInfo}(undef, size(X, 2))
    for feat in 1:size(X, 2)
        splits[feat] = SplitInfo(-Inf, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
    end

    tracks = Vector{SplitTrack}(undef, size(X, 2))
    for feat in 1:size(X, 2)
        tracks[feat] = SplitTrack(0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
    end

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = Int[]

        # grow nodes
        for id in active_id

            node = train_nodes[id]

            if tree_depth == params.max_depth
                push!(tree.nodes, TreeNode(- node.∑δ / (node.∑δ² + params.λ)))
            else
                node_size = size(node.𝑖, 1)

                @threads for feat in node.𝑗
                # for feat in node.𝑗
                    sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = false)
                    find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², params.λ, splits[feat], tracks[feat])
                    # find_split!(view(X, node.𝑖[perm_ini[1:node_size, feat]], feat), view(δ, node.𝑖[perm_ini[1:node_size, feat]]) , view(δ², node.𝑖[perm_ini[1:node_size, feat]]), node.∑δ, node.∑δ², params.λ, splits[feat], tracks[feat])
                    splits[feat].feat = feat
                end

                # assign best split
                best = get_max_gain(splits)
                # println(best)

                # grow node if best split improve gain
                if best.gain > node.gain + params.γ

                    # Node: depth, ∑δ, ∑δ², gain, 𝑖, 𝑗
                    # train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, view(node.𝑖, view(perm_ini, 1:best.𝑖, best.feat)), view(node.𝑗, :))
                    # train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, view(node.𝑖, view(perm_ini, best.𝑖+1:node_size, best.feat)), view(node.𝑗, :))
                    train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, node.𝑖[perm_ini[1:best.𝑖, best.feat]], node.𝑗[:])
                    train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, node.𝑖[perm_ini[best.𝑖+1:node_size, best.feat]], node.𝑗[:])

                    # push split Node
                    # push!(tree.nodes, SplitNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond))
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond))

                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)

                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode(- node.∑δ / (node.∑δ² + params.λ)))
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
function get_max_gain(splits)
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    # best.feat = feat
    return best
end


function grow_gbtree(X::AbstractArray{T, 2}, Y::AbstractArray{<:AbstractFloat, 1}, params::Params; X_eval::AbstractArray{T, 2} = Array{T, 2}(undef, (0,0)), Y_eval::AbstractArray{<:AbstractFloat, 1} = Array{Float64, 1}(undef, 0))  where T<:Real
    μ = mean(Y)
    # pred = ones(size(Y, 1)) .* μ
    @fastmath pred = ones(size(Y, 1)) .* μ

    δ, δ² = zeros(Float64, size(Y, 1)), zeros(Float64, size(Y, 1))
    update_grads!(Val{params.loss}(), pred, Y, δ, δ²)

    # eval init
    if size(Y_eval, 1) > 0
        # pred_eval = ones(size(Y_eval, 1)) .* μ
        @fastmath pred_eval = ones(size(Y_eval, 1)) .* μ
    end

    ∑δ, ∑δ² = sum(δ), sum(δ²)
    gain = get_gain(∑δ, ∑δ², params.λ)

    bias = TreeNode(μ)
    bias = Tree([bias])
    gbtree = GBTree([bias], params)

    # sort perm id placeholder
    perm_ini = zeros(Int, size(X))

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    train_nodes = Vector{TrainNode}(undef, 2^params.max_depth-1)
    for feat in 1:2^params.max_depth-1
        train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, [0], [0])
    end

    for i in 1:params.nrounds
        # select random rows and cols
        # 𝑖 = view(𝑖_, sample(𝑖_, floor(Int, params.rowsample * X_size[1]), replace = false))
        # 𝑗 = view(𝑗_, sample(𝑗_, floor(Int, params.colsample * X_size[2]), replace = false))
        𝑖 = 𝑖_[sample(𝑖_, floor(Int, params.rowsample * X_size[1]), replace = false)]
        𝑗 = 𝑗_[sample(𝑗_, floor(Int, params.colsample * X_size[2]), replace = false)]
        # 𝑖 = 𝑖_
        # 𝑗 = 𝑗_

        # get gradients
        update_grads!(Val{params.loss}(), pred, Y, δ, δ²)
        ∑δ, ∑δ² = sum(δ), sum(δ²)
        gain = get_gain(∑δ, ∑δ², params.λ)

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², gain, 𝑖, 𝑗)
        tree = grow_tree(X, δ, δ², params, perm_ini, train_nodes)

        # get update predictions
        predict!(pred, tree, X)
        # eval predictions
        if size(Y_eval, 1) > 0
            predict!(pred_eval, tree, X_eval)
        end
        # update push tree to model
        push!(gbtree.trees, tree)

        # callback function
        if mod(i, 10) == 0
            if size(Y_eval, 1) > 0
                println("iter:", i, ", train:", sqrt(mean((pred .- Y) .^ 2)), ", eval: ", sqrt(mean((pred_eval .- Y_eval) .^ 2)))
            else
                println("iter:", i, ", train:", sqrt(mean((pred .- Y) .^ 2)))
            end
        end # end of callback

    end #end of nrounds
    return gbtree
end


function find_split!(x::AbstractArray{T, 1}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, ∑δ, ∑δ², λ, info::SplitInfo, track::SplitTrack) where T<:Real

    # info.gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0
    @fastmath info.gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²

    @fastmath @inbounds for i in 1:(size(x, 1) - 1)
    # @inbounds for i in 1:(size(x, 1) - 1)

        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]

        @fastmath @inbounds if x[i] < x[i+1] # check gain only if there's a change in value
        # @inbounds if x[i] < x[i+1] # check gain only if there's a change in value

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
