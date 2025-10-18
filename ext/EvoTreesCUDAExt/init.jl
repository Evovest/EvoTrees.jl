function EvoTrees.init_core(params::EvoTrees.EvoTypes, ::Type{<:EvoTrees.GPU}, data, fnames, y_train, w, offset)

    edges, featbins, feattypes = EvoTrees.get_edges(data; feature_names=fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(EvoTrees.binarize(data; feature_names=fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32
    L = EvoTrees._loss2type_dict[params.loss]

    target_levels = nothing
    target_isordered = false
    if L == EvoTrees.LogLoss
        @assert eltype(y_train) <: Real && minimum(y_train) >= 0 && maximum(y_train) <= 1
        K = 1
        y = T.(y_train)
        μ = [EvoTrees.logit(EvoTrees.mean(y))]
        !isnothing(offset) && (offset .= EvoTrees.logit.(offset))
    elseif L in [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = fill(log(EvoTrees.mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.MLogLoss
        if eltype(y_train) <: EvoTrees.CategoricalValue
            target_levels = EvoTrees.CategoricalArrays.levels(y_train)
            target_isordered = EvoTrees.isordered(y_train)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(y_train))
        elseif eltype(y_train) <: Integer || eltype(y_train) <: Bool || eltype(y_train) <: String || eltype(y_train) <: Char
            target_levels = sort(unique(y_train))
            yc = EvoTrees.CategoricalVector(y_train, levels=target_levels)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(yc))
        else
            @error "Invalid target eltype: $(eltype(y_train))"
        end
        K = length(target_levels)
        μ = T.(log.(EvoTrees.proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.GaussianMLE
        @assert eltype(y_train) <: Real
        K = 2
        y = T.(y_train)
        μ = [EvoTrees.mean(y), log(EvoTrees.std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == EvoTrees.LogisticMLE
        @assert eltype(y_train) <: Real
        K = 2
        y = T.(y_train)
        μ = [EvoTrees.mean(y), log(EvoTrees.std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = [EvoTrees.mean(y)]
    end
    y = CuArray(y)
    μ = T.(μ)
    !isnothing(offset) && (μ .= 0)

    backend = KernelAbstractions.get_backend(x_bin)
    pred = KernelAbstractions.zeros(backend, T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    ∇ = KernelAbstractions.zeros(backend, T, 2 * K + 1, nobs)
    h∇ = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    h∇L = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    h∇R = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    nidx = KernelAbstractions.ones(backend, UInt32, nobs)
    is_in = KernelAbstractions.zeros(backend, UInt32, nobs)
    is_out = KernelAbstractions.zeros(backend, UInt32, nobs)
    mask = KernelAbstractions.zeros(backend, UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = KernelAbstractions.zeros(backend, UInt32, ceil(Int, params.colsample * nfeats))

    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :nrounds => 0,
        :feature_names => fnames,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    nodes = [EvoTrees.TrainNode(nfeats, params.nbins, K, view(zeros(UInt32, 0), 1:0)) for _ in 1:(2^params.max_depth-1)]
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(L, K, bias, info)

    cond_feats = zeros(Int, 2^(params.max_depth - 1) - 1)
    cond_bins = zeros(UInt8, 2^(params.max_depth - 1) - 1)
    cond_feats_gpu = CuArray(cond_feats)
    cond_bins_gpu = CuArray(cond_bins)
    feattypes_gpu = CuArray(feattypes)
    monotone_constraints_gpu = CuArray(monotone_constraints)

    max_nodes_level = 2^params.max_depth
    left_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    right_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    
    # FIX: Use correct tree node count
    max_tree_nodes = 2^params.max_depth - 1
    target_mask_buf = KernelAbstractions.zeros(backend, UInt8, max_tree_nodes)
    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, max_tree_nodes)
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt8, max_tree_nodes)
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_tree_nodes)
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, K, max_tree_nodes)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, max_tree_nodes)
    node_counts_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = KernelAbstractions.zeros(backend, Int32, 1)
    best_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    build_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    subtract_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    build_count = KernelAbstractions.zeros(backend, Int32, 1)
    subtract_count = KernelAbstractions.zeros(backend, Int32, 1)
    sums_temp_gpu = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, max_nodes_level)

    cache = CacheGPU(
        Dict(:nrounds => 0),
        x_bin,
        y,
        w,
        K,
        nodes,
        pred,
        nidx,
        is_in,
        is_out,
        mask,
        js_,
        js,
        ∇,
        h∇,
        h∇L,
        h∇R,
        fnames,
        edges,
        featbins,
        feattypes_gpu,
        cond_feats,
        cond_feats_gpu,
        cond_bins,
        cond_bins_gpu,
        monotone_constraints_gpu,
        left_nodes_buf,
        right_nodes_buf,
        target_mask_buf, tree_split_gpu,
        tree_cond_bin_gpu,
        tree_feat_gpu,
        tree_gain_gpu,
        tree_pred_gpu,
        nodes_sum_gpu,
        anodes_gpu,
        n_next_gpu,
        n_next_active_gpu,
        best_gain_gpu,
        best_bin_gpu,
        best_feat_gpu,
        build_nodes_gpu,
        subtract_nodes_gpu,
        build_count,
        subtract_count,
        node_counts_gpu,
        sums_temp_gpu,
    )

    return m, cache
end

