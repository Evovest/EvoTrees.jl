function grow_tree!(tree::Tree, X::AbstractArray{T, 2}, δ::AbstractArray{<:AbstractFloat, 1}, δ²::AbstractArray{<:AbstractFloat, 1}, params::Params, perm_ini::AbstractArray{Int}) where T<:Real

    active_id = [1]
    leaf_count = 1
    tree_depth = 1

    splits = Vector{SplitInfo2}(undef, size(X, 2))
    for feat in 1:size(X, 2)
        splits[feat] = SplitInfo2(-Inf, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
    end

    tracks = Vector{SplitTrack}(undef, size(X, 2))
    for feat in 1:size(X, 2)
        tracks[feat] = SplitTrack(0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
    end

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth < params.max_depth
        next_active_id = []
        # grow nodes
        for id in active_id
            node = tree.nodes[id]
            node_size = size(node.𝑖, 1)

            @threads for feat in 1:size(X, 2)
                sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = false)
                find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², params.λ, splits[feat], tracks[feat])
            end

            # assign best split
            best = get_max_gain(splits)

            # grow node if best split improve gain
            if best.gain > node.gain + params.γ

                # child nodes id
                node.left = leaf_count + 1
                node.right = leaf_count + 2
                # update list of next depth nodes
                node.feat = best.feat
                node.cond = best.cond

                # Node: depth, ∑δ, ∑δ², gain, feat, cond, left, right, pred, 𝑖 - for perm_id
                push!(tree.nodes, Node(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0, 0.0, 0, 0, - best.∑δL / (best.∑δ²L + params.λ) * params.η, view(node.𝑖, view(perm_ini, 1:best.𝑖, node.feat))))
                push!(tree.nodes, Node(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, 0, 0.0, 0, 0, - best.∑δR / (best.∑δ²R + params.λ) * params.η, view(node.𝑖, view(perm_ini, best.𝑖+1:node_size, node.feat))))

                # update list of active nodes for next depth
                if node.depth + 1 < params.max_depth
                    if best.𝑖 > params.min_weight
                        push!(next_active_id, leaf_count + 1)
                    end
                    if node_size - best.𝑖 > params.min_weight
                        push!(next_active_id, leaf_count + 2)
                    end
                end
                leaf_count += 2
            # else # action if no split found
            end # end of single node split search
            # node.𝑖 = [0]
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits)
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    best.feat = feat
    return best
end


function grow_gbtree(X::AbstractArray{T, 2}, Y::AbstractArray{<:AbstractFloat, 1}, params::Params; X_eval::AbstractArray{T, 2} = Array{T, 2}(undef, (0,0)), Y_eval::AbstractArray{<:AbstractFloat, 1} = Array{Float64, 1}(undef, 0))  where T<:Real
    μ = mean(Y)
    pred = ones(size(Y, 1)) .* μ
    δ, δ² = zeros(Float64, size(Y, 1)), zeros(Float64, size(Y, 1))
    update_grads!(Val{params.loss}(), pred, Y, δ, δ²)

    # eval init
    if size(Y_eval, 1) > 0
        pred_eval = ones(size(Y_eval, 1)) .* μ
    end

    ∑δ, ∑δ² = sum(δ), sum(δ²)
    gain = get_gain(∑δ, ∑δ², params.λ)

    bias = Node(1, 0.0, 0.0, gain, 0, 0.0, 0, 0, μ, collect(1:size(X,1)))
    bias = Tree([bias])
    gbtree = GBTrees([bias], params)

    # sort perm id placeholder
    perm_ini = zeros(Int, size(X))

    X_size = size(X)
    𝑖 = collect(1:X_size[1])
    𝑗 = collect(1:X_size[2])

    for i in 1:params.nrounds
        # select random rows and cols
        # 𝑖 = view(𝑖, sample(𝑖, floor(Int, params.rowsample * X_size[1]), replace = false))
        # 𝑗 = view(𝑗, sample(𝑗, floor(Int, params.colsample * X_size[2]), replace = false))
        # get gradients
        update_grads!(Val{params.loss}(), pred, Y, δ, δ²)
        ∑δ, ∑δ² = sum(δ), sum(δ²)
        gain = get_gain(∑δ, ∑δ², params.λ)

        # assign a root and grow tree
        # tree = Tree([Node(1, ∑δ, ∑δ², gain, 0, 0.0, 0, 0, - ∑δ / (∑δ² + params.λ), view(𝑖, :))])
        tree = Tree([Node(1, ∑δ, ∑δ², gain, 0, 0.0, 0, 0, - ∑δ / (∑δ² + params.λ), 𝑖)])
        grow_tree!(tree, view(X, :, :), view(δ, :), view(δ², :), params, view(perm_ini, :, :))
        # grow_tree!(tree, X, δ, δ², params, perm_ini)
        # grow_tree!(tree, X[𝑖, 𝑗], δ[𝑖], δ²[𝑖], params, perm_ini[𝑖, 𝑗])
        # grow_tree!(tree, view(X, 𝑖, 𝑗), view(δ, 𝑖), view(δ², 𝑖), params, view(perm_ini, 𝑖, 𝑗))
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
                println("iter:", i, ", train:", mean((pred .- Y) .^ 2), ", eval: ", mean((pred_eval .- Y_eval) .^ 2))
            else
                println("iter:", i, ", train:", mean((pred .- Y) .^ 2))
            end
        end # end of callback

    end #end of nrounds
    return gbtree
end



function find_split!(x::AbstractArray{T, 1}, δ::AbstractArray{<:AbstractFloat, 1}, δ²::AbstractArray{<:AbstractFloat, 1}, ∑δ, ∑δ², λ, info::SplitInfo2, track::SplitTrack) where T<:Real

    info.gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²

    𝑖 = 1
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
