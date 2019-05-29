# initialize train_nodes
function grow_tree(X::AbstractArray{R, 2}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}, params::EvoTreeRegressor, perm_ini::AbstractArray{Int}, train_nodes::Vector{TrainNode{T, I, J, S}}, splits::Vector{SplitInfo{Float64, Int}}, tracks::Vector{SplitTrack{Float64}}, X_edges) where {R<:Real, T<:AbstractFloat, I<:AbstractArray{Int, 1}, J<:AbstractArray{Int, 1}, S<:Int}

    active_id = ones(Int, 1)
    leaf_count = 1::Int
    tree_depth = 1::Int
    tree = Tree(Vector{TreeNode{Float64, Int, Bool}}())

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑𝑤 <= params.min_weight
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, view(δ², node.𝑖))))
            else
                node_size = size(node.𝑖, 1)
                @threads for feat in node.𝑗
                    sortperm!(view(perm_ini, 1:node_size, feat), view(X, node.𝑖, feat), alg = QuickSort, initialized = false)
                    find_split!(view(X, view(node.𝑖, view(perm_ini, 1:node_size, feat)), feat), view(δ, view(node.𝑖, view(perm_ini, 1:node_size, feat))) , view(δ², view(node.𝑖, view(perm_ini, 1:node_size, feat))), view(𝑤, view(node.𝑖, view(perm_ini, 1:node_size, feat))), node.∑δ, node.∑δ², node.∑𝑤, params, splits[feat], tracks[feat], X_edges[feat])
                end
                # assign best split
                best = get_max_gain(splits)
                # grow node if best split improve gain
                if best.gain > node.gain + params.γ
                    train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.∑𝑤L, best.gainL, node.𝑖[perm_ini[1:best.𝑖, best.feat]], node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.∑𝑤R, best.gainR, node.𝑖[perm_ini[best.𝑖+1:node_size, best.feat]], node.𝑗)
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond))
                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)
                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, view(δ², node.𝑖))))
                end # end of single node split search
            end
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits::Vector{SplitInfo{Float64,Int}})
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    # best.feat = feat
    return best
end

function get_edges(X, nbins=250)
    edges = Vector{Vector}(undef, size(X,2))
    @threads for i in 1:size(X, 2)
        edges[i] = unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:(end-1)]
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :,i))]
        end
    end
    return edges
end

function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i in 1:size(X, 2)
        X_bin[:,i] = searchsortedlast.(Ref(edges[i]), view(X,:,i)) .+ 1
    end
    X_bin
end

# grow_gbtree
function grow_gbtree(X::AbstractArray{R, 2}, Y::AbstractArray{T, 1}, params::EvoTreeRegressor;
    X_eval::AbstractArray{R, 2} = Array{R, 2}(undef, (0,0)), Y_eval::AbstractArray{T, 1} = Array{Float64, 1}(undef, 0),
    early_stopping_rounds=Int(1e5), print_every_n=100, verbosity=1) where {R<:Real, T<:AbstractFloat}

    X_edges = get_edges(X, params.nbins)
    X_bin = binarize(X, X_edges)

    μ = mean(Y)
    if typeof(params.loss) == Logistic
        μ = logit(μ)
    elseif params.loss == Poisson
        μ = log(μ)
    end
    pred = ones(size(Y, 1)) .* μ

    # initialize gradients and weights
    δ, δ² = zeros(Float64, size(Y, 1)), zeros(Float64, size(Y, 1))
    𝑤 = ones(Float64, size(Y, 1))

    # eval init
    if size(Y_eval, 1) > 0
        pred_eval = ones(size(Y_eval, 1)) .* μ
    end

    bias = Tree([TreeNode(μ)])
    gbtree = GBTree([bias], params, Metric())

    # sort perm id placeholder
    perm_ini = zeros(Int, size(X_bin))

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize train nodes
    train_nodes = Vector{TrainNode{Float64, Array{Int64,1}, Array{Int64, 1}, Int64}}(undef, 2^params.max_depth-1)
    for feat in 1:2^params.max_depth-1
        train_nodes[feat] = TrainNode(0, -Inf, -Inf, -Inf, -Inf, [0], [0])
    end

    # initialize metric
    if params.metric != :none
        metric_track = Metric()
        metric_best = Metric()
        iter_since_best = 0
    end

    # loop over nrounds
    for i in 1:params.nrounds
        # select random rows and cols
        𝑖 = 𝑖_[sample(𝑖_, ceil(Int, params.rowsample * X_size[1]), replace = false)]
        𝑗 = 𝑗_[sample(𝑗_, ceil(Int, params.colsample * X_size[2]), replace = false)]

        # get gradients
        update_grads!(params.loss, params.α, pred, Y, δ, δ², 𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

        # initializde node splits info and tracks - colsample size (𝑗)
        splits = Vector{SplitInfo{Float64, Int64}}(undef, X_size[2])
        for feat in 𝑗_
            splits[feat] = SplitInfo{Float64, Int64}(-Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, feat, 0.0)
        end
        tracks = Vector{SplitTrack{Float64}}(undef, X_size[2])
        for feat in 𝑗_
            tracks[feat] = SplitTrack{Float64}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf)
        end

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(X_bin, δ, δ², 𝑤, params, perm_ini, train_nodes, splits, tracks, X_edges)
        # update push tree to model
        push!(gbtree.trees, tree)

        # get update predictions
        predict!(pred, tree, X)
        # eval predictions
        if size(Y_eval, 1) > 0
            predict!(pred_eval, tree, X_eval)
        end

        # callback function
        if params.metric != :none

            if size(Y_eval, 1) > 0
                metric_track.metric .= eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric .= eval_metric(Val{params.metric}(), pred, Y, params.α)
            end

            if metric_track.metric < metric_best.metric
                metric_best.metric .=  metric_track.metric
                metric_best.iter .=  i
            else
                iter_since_best += 1
            end

            if mod(i, print_every_n) == 0 && verbosity > 0
                display(string("iter:", i, ", eval: ", metric_track.metric))
            end
            iter_since_best >= early_stopping_rounds ? break : nothing
        end
    end #end of nrounds

    if params.metric != :none
        gbtree.metric.iter .= metric_best.iter
        gbtree.metric.metric .= metric_best.metric
    end
    return gbtree
end

# find best split
function find_split!(x::AbstractArray{T, 1}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, 𝑤::AbstractArray{Float64, 1}, ∑δ, ∑δ², ∑𝑤, params::EvoTreeRegressor, info::SplitInfo, track::SplitTrack, x_edges) where T<:Real

    info.gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑𝑤L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    @inbounds for i in 1:(size(x, 1) - 1)
    # @fastmath @inbounds for i in eachindex(x)

        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑𝑤L += 𝑤[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]
        track.∑𝑤R -= 𝑤[i]

        @inbounds if x[i] < x[i+1] && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight # check gain only if there's a change in value
        # @inbounds if x[i] < x[i+1] # check gain only if there's a change in value

            update_track!(params.loss, track, params.λ)
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
                info.cond = x_edges[x[i]]
                info.𝑖 = i
            end
        end
    end
end
