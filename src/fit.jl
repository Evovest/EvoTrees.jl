# initialise evotree
function init_evotree(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; fnames=nothing, verbosity=1) where {T,U,S}

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)
    if typeof(params.loss) == Logistic
        Y = T.(Y)
        μ = [logit(mean(Y))]
    elseif typeof(params.loss) == Poisson
        Y = T.(Y)
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Softmax
        if eltype(Y) <: CategoricalValue
            levels = CategoricalArrays.levels(Y)
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(CategoricalArrays.levelcode.(Y))
        else
            levels = sort(unique(Y))
            yc = CategoricalVector(Y, levels=levels)
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y = T.(Y)
        μ = SVector{2}([mean(Y), log(std(Y))])
    else
        Y = T.(Y)
        μ = [mean(Y)]
    end

    # initialize preds
    X_size = size(X)
    pred_cpu = zeros(T, X_size[1], K)
    @inbounds for i in eachindex(pred_cpu)
        pred_cpu[i,:] .= μ
    end

    bias = Tree([TreeNode(SVector{K,T}(μ))])
    evotree = GBTree([bias], params, Metric(), K, levels)

    𝑖_ = UInt32.(collect(1:X_size[1]))
    𝑗_ = UInt32.(collect(1:X_size[2]))

    # initialize gradients and weights
    δ = ones(T, X_size[1], 2 * K + 1)

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initializde histograms
    hist = zeros(T, 2 * K + 1, params.nbins, X_size[2], 2^params.max_depth - 1)
    
    # initialize train nodes
    train_nodes = Vector{TrainNode{T,UInt32,Vector{T}}}(undef, 2^params.max_depth - 1)

    cache = (params = deepcopy(params),
        X = X, Y_cpu = Y, K = K,
        pred_cpu = pred_cpu,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, δ = δ,
        edges = edges, 
        X_bin = X_bin,
        train_nodes = train_nodes,
        hist = hist)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree{L,T,S}, cache; verbosity=1) where {L,T,S}

    # initialize from cache
    params = evotree.params
    train_nodes = cache.train_nodes
    splits = cache.splits
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = cache.𝑖_[sample(params.rng, cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = cache.𝑗_[sample(params.rng, cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]
        # reset gain to -Inf
        for feat in cache.𝑗_
            splits[feat].gain = T(-Inf)
        end

        # build a new tree
        update_grads!(params.loss, params.α, cache.pred_cpu, cache.Y_cpu, cache.δ, cache.δ², cache.𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(cache.δ[𝑖]), sum(cache.δ²[𝑖]), sum(cache.𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)
        # assign a root and grow tree
        train_nodes[1] = TrainNode(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.δ², cache.𝑤, cache.hist_δ, cache.hist_δ², cache.hist_𝑤, params, train_nodes, splits, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        predict!(cache.pred_cpu, tree, cache.X)

    end # end of nrounds

    cache.params.nrounds = params.nrounds
    return evotree
end

# grow a single tree
function grow_tree(δ, δ², 𝑤,
    hist_δ, hist_δ², hist_𝑤,
    params::EvoTypes{T,U,S},
    train_nodes::Vector{TrainNode{L,T,S}},
    splits::Vector{SplitInfo{L,T,Int}},
    edges, X_bin) where {T <: AbstractFloat,U,S,L}

    active_id = ones(Int, 1)
    leaf_count = one(Int)
    tree_depth = one(Int)
    tree = Tree(Vector{TreeNode{L,T,Int,Bool}}())

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑𝑤[1] <= params.min_weight + 1e-8
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
            else
                if id > 1 && id == tree.nodes[node.parent].right
                    # println("id is right:", id)
                    hist_δ[id] .= hist_δ[node.parent] .- hist_δ[id - 1]
                    hist_δ²[id] .= hist_δ²[node.parent] .- hist_δ²[id - 1]
                    hist_𝑤[id] .= hist_𝑤[node.parent] .- hist_𝑤[id - 1]
                else
                    # println("id is left:", id)
                    update_hist!(hist_δ[id], hist_δ²[id], hist_𝑤[id], δ, δ², 𝑤, X_bin, node)
                end
                for j in node.𝑗
                    splits[j].gain = node.gain
                    find_split!(view(hist_δ[id], :, j), view(hist_δ²[id], :, j), view(hist_𝑤[id], :, j), params, node, splits[j], edges[j])
                end

                best = get_max_gain(splits)
                # grow node if best split improves gain
                if best.gain > node.gain + params.γ
                    left, right = update_set(node.𝑖, best.𝑖, view(X_bin, :, best.feat))
                    # println("id: ∑𝑤/length(node/left/right) / ", id, " : ", node.∑𝑤, " / ", length(node.𝑖), " / ", length(left), " / ", length(right), " / ", best.𝑖)
                    train_nodes[leaf_count + 1] = TrainNode(id, node.depth + 1, best.∑δL, best.∑δ²L, best.∑𝑤L, best.gainL, left, node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNode(id, node.depth + 1, best.∑δR, best.∑δ²R, best.∑𝑤R, best.gainR, right, node.𝑗)
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond, best.gain - node.gain, L))
                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)
                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
                end # end of single node split search
            end
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits::Vector{SplitInfo{L,T,S}}) where {L,T,S}
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    return best
end

function fit_evotree(params, X_train, Y_train;
    X_eval=nothing, Y_eval=nothing,
    early_stopping_rounds=9999,
    eval_every_n=1,
    print_every_n=9999,
    verbosity=1)

    # initialize metric
    iter_since_best = 0
    if params.metric != :none
        metric_track = Metric()
        metric_best = Metric()
    end

    nrounds_max = params.nrounds
    params.nrounds = 0
    if params.device == "gpu"
        model, cache = init_evotree_gpu(params, X_train, Y_train)
    else 
        model, cache = init_evotree(params, X_train, Y_train)
    end

    iter = 1
    if params.metric != :none && X_eval !== nothing
        pred_eval = predict(model.trees[1], X_eval, model.K)
        Y_eval = convert.(eltype(cache.Y_cpu), Y_eval)
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree!(model, cache)
        # callback function
        if params.metric != :none
            if X_eval !== nothing
                predict!(pred_eval, model.trees[model.params.nrounds + 1], X_eval)
                metric_track.metric = eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric = eval_metric(Val{params.metric}(), cache.pred_cpu, cache.Y_cpu, params.α)
            end
            if metric_track.metric < metric_best.metric
                metric_best.metric = metric_track.metric
                metric_best.iter =  model.params.nrounds
                iter_since_best = 0
            else
                iter_since_best += 1
            end
            if mod(model.params.nrounds, print_every_n) == 0 && verbosity > 0
                display(string("iter:", model.params.nrounds, ", eval: ", metric_track.metric))
            end
        end # end of callback
    end
    if params.metric != :none
        model.metric.iter = metric_best.iter
        model.metric.metric = metric_best.metric
    end
    params.nrounds = nrounds_max
    return model
end
