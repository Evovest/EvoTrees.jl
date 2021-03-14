# initialise evotree
function init_evotree_gpu(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; verbosity=1) where {T,U,S}

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)
    if typeof(params.loss) == Logistic
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [logit(mean(Y))]
    elseif typeof(params.loss) == Poisson
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = fill(log(mean(Y)), 1)
    elseif typeof(params.loss) == Softmax
        if typeof(Y) <: AbstractCategoricalVector
            levels = CategoricalArray(CategoricalArrays.levels(Y))
            K = length(levels)
            μ = zeros(T, K)
            Y = MLJModelInterface.int.(Y)
        else
            levels = CategoricalArray(sort(unique(Y)))
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(Y)
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [mean(Y), log(std(Y))]
    else
        Y_cpu = T.(Y)
        Y = CuArray(Y_cpu)
        μ = [mean(Y)]
    end

    # initialize preds
    X_size = size(X)
    pred_cpu = zeros(T, X_size[1], K)
    pred = CUDA.zeros(T, X_size[1], K)
    pred_cpu .= μ'
    pred .= CuArray(μ)'

    bias = Tree_gpu([TreeNode_gpu(μ)])
    evotree = GBTree_gpu([bias], params, Metric(), K, levels)

    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize gradients and weights
    δ, δ² = CUDA.zeros(T, X_size[1], K), CUDA.zeros(T, X_size[1], K)
    𝑤 = CUDA.ones(T, X_size[1])

    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin_cpu = binarize(X, edges)
    X_bin = CuArray(X_bin_cpu)

    # initializde histograms
    hist_δ = [CUDA.zeros(T, params.nbins, K, X_size[2]) for i in 1:2^params.max_depth-1]
    hist_δ² = [CUDA.zeros(T, params.nbins, K, X_size[2]) for i in 1:2^params.max_depth-1]
    hist_𝑤 = [CUDA.zeros(T, params.nbins, X_size[2]) for i in 1:2^params.max_depth-1]

    # initialize train nodes
    train_nodes = Vector{TrainNode_gpu{T, S, CuVector{S}}}(undef, 2^params.max_depth-1)
    # for node in 1:2^params.max_depth-1
    #     train_nodes[node] = TrainNode_gpu(0, 0, zeros(T, K), zeros(T, K), T(-Inf), T(-Inf), [0], [0])
    # end

    splits = Vector{SplitInfo_gpu{T, Int64}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo_gpu{T, Int}(T(-Inf), zeros(T,K), zeros(T,K), zero(T), zeros(T,K), zeros(T,K), zero(T), T(-Inf), T(-Inf), 0, feat, 0.0)
    end

    cache = (params=deepcopy(params),
        X=X, Y=Y, Y_cpu=Y_cpu, K=K,
        pred=pred, pred_cpu=pred_cpu,
        𝑖_=𝑖_, 𝑗_=𝑗_, δ=δ, δ²=δ², 𝑤=𝑤,
        edges=edges,
        X_bin=X_bin,
        train_nodes=train_nodes, splits=splits,
        hist_δ=hist_δ, hist_δ²=hist_δ², hist_𝑤=hist_𝑤)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree_gpu!(evotree::GBTree_gpu{T,S}, cache; verbosity=1) where {T,S}

    # initialize from cache
    params = evotree.params
    train_nodes = cache.train_nodes
    splits = cache.splits
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = CuVector(cache.𝑖_[sample(params.rng, cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)])
        𝑗 = CuVector(cache.𝑗_[sample(params.rng, cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)])
        # reset gain to -Inf
        for feat in cache.𝑗_
            splits[feat].gain = T(-Inf)
        end

        # build a new tree
        update_grads_gpu!(params.loss, cache.δ, cache.δ², cache.pred, cache.Y, cache.𝑤)
        # sum Gradients of each of the K parameters and bring to CPU
        ∑δ, ∑δ², ∑𝑤 = Vector(vec(sum(cache.δ[𝑖,:], dims=1))), Vector(vec(sum(cache.δ²[𝑖,:], dims=1))), sum(cache.𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)
        # # assign a root and grow tree
        train_nodes[1] = TrainNode_gpu(0, 1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree_gpu(cache.δ, cache.δ², cache.𝑤, cache.hist_δ, cache.hist_δ², cache.hist_𝑤, params, cache.K, train_nodes, splits, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        # bad GPU usage - to be imprived!
        predict_gpu!(cache.pred_cpu, tree, cache.X)
        cache.pred .= CuArray(cache.pred_cpu)

    end #end of nrounds

    cache.params.nrounds = params.nrounds
    # cache = (deepcopy(params), X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤)
    # return model, cache
    return evotree
end

# grow a single tree
function grow_tree_gpu(δ, δ², 𝑤,
    hist_δ, hist_δ², hist_𝑤,
    params::EvoTypes{T,U,S}, K,
    train_nodes::Vector{TrainNode_gpu{T,S,V}},
    splits::Vector{SplitInfo_gpu{T,Int}},
    edges, X_bin) where {T,U,S,V}

    active_id = ones(Int, 1)
    leaf_count = one(Int)
    tree_depth = one(Int)
    tree = Tree_gpu(Vector{TreeNode_gpu{T, Int, Bool}}())

    hist_δ_cpu = zeros(T, size(hist_δ[1]))
    hist_δ²_cpu = zeros(T, size(hist_δ²[1]))
    hist_𝑤_cpu = zeros(T, size(hist_𝑤[1]))

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑𝑤 <= params.min_weight + 1e-12 # rounding needed from histogram substraction
                push!(tree.nodes, TreeNode_gpu(pred_leaf_gpu(params.loss, node, params, δ²)))
            else
                if id > 1 && id == tree.nodes[node.parent].right
                    # println("id is right:", id)
                    hist_δ[id] .= hist_δ[node.parent] .- hist_δ[id-1]
                    hist_δ²[id] .= hist_δ²[node.parent] .- hist_δ²[id-1]
                    hist_𝑤[id] .= hist_𝑤[node.parent] .- hist_𝑤[id-1]
                else
                    # println("id is left:", id)
                    # should revisite to launch all hist update within depth once since async - and then
                    update_hist_gpu!(hist_δ[id], hist_δ²[id], hist_𝑤[id], δ, δ², 𝑤, X_bin, node.𝑖, node.𝑗, K)
                end

                hist_δ_cpu .= hist_δ[id]
                hist_δ²_cpu .= hist_δ²[id]
                hist_𝑤_cpu .= hist_𝑤[id]

                for j in Array(node.𝑗)
                    splits[j].gain = node.gain
                    find_split_gpu!(view(hist_δ_cpu,:,:,j), view(hist_δ²_cpu,:,:,j), view(hist_𝑤_cpu,:,j), params, node, splits[j], edges[j])
                end

                best = get_max_gain_gpu(splits)

                # grow node if best split improves gain
                if best.gain > node.gain + params.γ
                    left, right = update_set_gpu(node.𝑖, best.𝑖, view(X_bin,:,best.feat))

                    # println("id: ∑𝑤/length(node/left/right) / ", id, " : ", node.∑𝑤, " / ", length(node.𝑖), " / ", length(left), " / ", length(right), " / ", best.𝑖)
                    train_nodes[leaf_count + 1] = TrainNode_gpu(id, node.depth + 1, copy(best.∑δL), copy(best.∑δ²L), best.∑𝑤L, best.gainL, left, node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNode_gpu(id, node.depth + 1, copy(best.∑δR), copy(best.∑δ²R), best.∑𝑤R, best.gainR, right, node.𝑗)
                    push!(tree.nodes, TreeNode_gpu(leaf_count + 1, leaf_count + 2, best.feat, best.cond, best.gain-node.gain, K))
                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)
                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode_gpu(pred_leaf_gpu(params.loss, node, params, δ²)))
                end # end of single node split search
            end
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain_gpu(splits::Vector{SplitInfo_gpu{T,S}}) where {T,S}
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    return best
end

function fit_evotree_gpu(params, X_train, Y_train;
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
    model, cache = init_evotree_gpu(params, X_train, Y_train)
    iter = 1

    if params.metric != :none && X_eval !== nothing
        pred_eval = predict_gpu(model.trees[1], X_eval, model.K)
        Y_eval = convert.(eltype(cache.Y), Y_eval)
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree_gpu!(model, cache)
        # callback function
        if params.metric != :none
            if X_eval !== nothing
                predict_gpu!(pred_eval, model.trees[model.params.nrounds+1], X_eval)
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
