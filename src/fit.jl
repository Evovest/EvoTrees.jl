# initialise evotree
function init_evotree(params::Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian},
    X::AbstractMatrix{R}, Y::AbstractVector{S}; verbosity=1) where {R<:Real, S}

    seed!(params.seed)

    display("start init")

    K = 1
    levels = ""
    if typeof(params.loss) == Logistic
        μ = fill(logit(mean(Y)), 1)
    elseif typeof(params.loss) == Poisson
        Y = Float64.(Y)
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Softmax
        if typeof(Y) <: AbstractCategoricalVector
            levels = CategoricalArray(CategoricalArrays.levels(Y))
            K = length(levels)
            μ = zeros(K)
            Y = MLJBase.int(Y)
        else
            levels = CategoricalArray(sort(unique(Y)))
            K = length(levels)
            μ = zeros(K)
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        μ = SVector{2}([mean(Y), log(var(Y))])
    else
        μ = fill(mean(Y), 1)
    end

    # initialize preds
    pred = zeros(SVector{K,Float64}, size(X,1))
    for i in eachindex(pred)
        pred[i] += μ
    end

    # bias = Tree([TreeNode(SVector{1, Float64}(μ))])
    bias = Tree([TreeNode(SVector{K,Float64}(μ))])
    evotree = GBTree([bias], params, Metric(), K, levels)

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    display("initialize gradients")
    # initialize gradients and weights
    δ, δ² = zeros(SVector{evotree.K, Float64}, X_size[1]), zeros(SVector{evotree.K, Float64}, X_size[1])
    # δ, δ² = zeros(SVector{1, Float64}, X_size[1]), zeros(SVector{1, Float64}, X_size[1])
    𝑤 = zeros(SVector{1, Float64}, X_size[1]) .+ one(Float64)
    display("gradients initialized")

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initialize train nodes
    train_nodes = Vector{TrainNode{evotree.K, Float64, Int64}}(undef, 2^params.max_depth-1)
    for node in 1:2^params.max_depth-1
        train_nodes[node] = TrainNode(0, SVector{evotree.K, Float64}(fill(-Inf, evotree.K)), SVector{evotree.K, Float64}(fill(-Inf, evotree.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, [0], [0])
    end

    # initializde node splits info and tracks - colsample size (𝑗)
    splits = Vector{SplitInfo{evotree.K, Float64, Int64}}(undef, X_size[2])
    hist_δ = Vector{Vector{SVector{evotree.K, Float64}}}(undef, X_size[2])
    hist_δ² = Vector{Vector{SVector{evotree.K, Float64}}}(undef, X_size[2])
    hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo{evotree.K, Float64, Int}(-Inf, SVector{evotree.K, Float64}(zeros(evotree.K)), SVector{evotree.K, Float64}(zeros(evotree.K)), SVector{1, Float64}(zeros(1)), SVector{evotree.K, Float64}(zeros(evotree.K)), SVector{evotree.K, Float64}(zeros(evotree.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
        hist_δ[feat] = zeros(SVector{evotree.K, Float64}, length(edges[feat]))
        hist_δ²[feat] = zeros(SVector{evotree.K, Float64}, length(edges[feat]))
        hist_𝑤[feat] = zeros(SVector{1, Float64}, length(edges[feat]))
    end

    display("before cache")

    cache = (params=deepcopy(params),
        X=X, Y=Y, pred=pred,
        𝑖_=𝑖_, 𝑗_=𝑗_, δ=δ, δ²=δ², 𝑤=𝑤,
        edges=edges, X_bin=X_bin,
        train_nodes=train_nodes, splits=splits,
        hist_δ=hist_δ, hist_δ²=hist_δ², hist_𝑤=hist_𝑤)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree, cache; verbosity=1)

    # initialize from cache
    params = evotree.params
    train_nodes = cache.train_nodes
    splits = cache.splits
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = cache.𝑖_[sample(cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = cache.𝑗_[sample(cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]
        # reset gain to -Inf
        for feat in cache.𝑗_
            splits[feat].gain = -Inf
        end

        # build a new tree
        update_grads!(params.loss, params.α, cache.pred, cache.Y, cache.δ, cache.δ², cache.𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(cache.δ[𝑖]), sum(cache.δ²[𝑖]), sum(cache.𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)
        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.δ², cache.𝑤, cache.hist_δ, cache.hist_δ², cache.hist_𝑤, params, train_nodes, splits, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        predict!(cache.pred, tree, cache.X)

    end #end of nrounds

    cache.params.nrounds = params.nrounds
    # cache = (deepcopy(params), X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤)
    # return model, cache
    return evotree
end

# grow a single tree
function grow_tree(δ, δ², 𝑤,
    hist_δ, hist_δ², hist_𝑤,
    params::Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian},
    train_nodes::Vector{TrainNode{L,T,S}},
    splits::Vector{SplitInfo{L,T,Int}},
    edges, X_bin) where {R<:Real, T<:AbstractFloat, S<:Int, L}

    active_id = ones(Int, 1)
    leaf_count = one(Int)
    tree_depth = one(Int)
    tree = Tree(Vector{TreeNode{L, T, Int, Bool}}())

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑𝑤[1] <= params.min_weight
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
            else
                @threads for feat in node.𝑗
                    splits[feat].gain = node.gain
                    find_split_static!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], view(X_bin,:,feat), δ, δ², 𝑤, node.∑δ, node.∑δ², node.∑𝑤, params, splits[feat], edges[feat], node.𝑖)
                    # update_hist!(hist_δ, hist_δ², hist_𝑤, X_bin, δ, δ², 𝑤, set, feat)
                    # find_split!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], node.∑δ, node.∑δ², node.∑𝑤, params, splits[feat], edges[feat], feat)
                end
                # assign best split
                best = get_max_gain(splits)
                # grow node if best split improve gain
                if best.gain > node.gain + params.γ
                    left, right = update_set(node.𝑖, best.𝑖, view(X_bin,:,best.feat))
                    train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.∑𝑤L, best.gainL, left, node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.∑𝑤R, best.gainR, right, node.𝑗)
                    # push split Node
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond, L))
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
    model, cache = init_evotree(params, X_train, Y_train)
    iter = 1

    if params.metric != :none && !isnothing(X_eval)
        pred_eval = predict(model.trees[1], X_eval, model.K)
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree!(model, cache)
        # callback function
        if params.metric != :none
            if !isnothing(X_eval)
                predict!(pred_eval, model.trees[model.params.nrounds+1], X_eval)
                metric_track.metric = eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric = eval_metric(Val{params.metric}(), cache.pred, Y_train, params.α)
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
