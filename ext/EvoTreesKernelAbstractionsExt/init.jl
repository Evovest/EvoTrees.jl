function EvoTrees.init_core(params::EvoTrees.EvoTypes, device::Type{<:EvoTrees.GPU}, data, feature_names, y_train, w, offset, group=nothing)

    rng = Xoshiro(params.seed)
    edges, featbins, feattypes = EvoTrees.get_edges(data; feature_names, nbins=params.nbins, rng)
    backend = _gpu_backend(device)
    x_bin = _to_device(backend, EvoTrees.binarize(data; feature_names, edges))
    nobs, nfeats = size(x_bin)
    T = Float32
    L = EvoTrees._loss2type_dict[params.loss]

    K, y_cpu, μ, target_levels, target_isordered = EvoTrees._init_target(L, y_train, params, offset, T)
    y = _to_device(backend, y_cpu)
    μ = T.(μ)
    !isnothing(offset) && (μ .= 0)

    pred = KernelAbstractions.zeros(backend, T, K, nobs)
    pred .= _to_device(backend, μ)
    !isnothing(offset) && (pred .+= _to_device(backend, collect(offset')))

    ∇ = KernelAbstractions.zeros(backend, T, 2 * K + 1, nobs)
    h∇ = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    @assert (size(y, ndims(y)) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    nidx = KernelAbstractions.ones(backend, UInt32, nobs)
    is_full = _to_device(backend, collect(UInt32, 1:nobs))
    mask_cpu = zeros(UInt8, nobs)
    mask_gpu = KernelAbstractions.zeros(backend, UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    n_sampled_feats = max(1, ceil(Int, params.colsample * nfeats))
    js = KernelAbstractions.zeros(backend, UInt32, n_sampled_feats)

    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    nodes = [EvoTrees.TrainNode(nfeats, params.nbins, K, view(zeros(UInt32, 0), 1:0)) for _ in 1:(2^(params.max_depth + 1) - 1)]
    m = EvoTree{L,K}(L, K, μ, info)

    cond_feats = zeros(UInt32, 2^params.max_depth - 1)
    cond_bins = zeros(UInt8, 2^params.max_depth - 1)
    cond_feats_gpu = _to_device(backend, cond_feats)
    cond_bins_gpu = _to_device(backend, cond_bins)
    feattypes_gpu = _to_device(backend, feattypes)
    monotone_constraints_gpu = _to_device(backend, monotone_constraints)

    max_tree_nodes = 2^(params.max_depth + 1) - 1
    left_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    right_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)

    target_mask_buf = KernelAbstractions.zeros(backend, UInt8, max_tree_nodes)
    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, max_tree_nodes)
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt8, max_tree_nodes)
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_tree_nodes)
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, K, max_tree_nodes)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, max_tree_nodes)
    node_counts_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)

    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    n_next_active_gpu = KernelAbstractions.zeros(backend, Int32, 1)
    best_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_tree_nodes)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    build_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    subtract_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    build_count = KernelAbstractions.zeros(backend, Int32, 1)
    subtract_count = KernelAbstractions.zeros(backend, Int32, 1)
    sums_temp_gpu = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, max_tree_nodes)

    n_sampled_feats = max(1, ceil(Int, params.colsample * nfeats))
    gains_per_feat_gpu = KernelAbstractions.zeros(backend, Float64, n_sampled_feats, max_tree_nodes)
    bins_per_feat_gpu = KernelAbstractions.zeros(backend, Int32, n_sampled_feats, max_tree_nodes)

    # Per-(node,feature) temp buffer for split scanning (K>1).
    # Layout: [2K+1, n_sampled_feats * max_tree_nodes]; col = (node_idx-1)*n_sampled_feats + feat_idx.
    split_sums_temp_gpu = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, n_sampled_feats * max_tree_nodes)

    # Oblivious level-gain buffers: sum over nodes per (bin, sampled feature).
    obliv_gains_gpu = KernelAbstractions.zeros(backend, Float64, params.nbins, n_sampled_feats)
    obliv_count_gpu = KernelAbstractions.zeros(backend, Int32, params.nbins, n_sampled_feats)

    Y = typeof(y)
    N = typeof(first(nodes))
    group_cache = if isnothing(group)
        nothing
    else
        ng = EvoTrees.ngroups(group)
        GroupCacheGPU(
            group,
            _to_device(backend, group.group),
            zeros(UInt8, ng),
            KernelAbstractions.zeros(backend, UInt8, ng),
        )
    end
    G = typeof(group_cache)

    cache = CacheBaseGPU{Y,N,G}(
        rng,
        K,
        x_bin,
        y,
        w,
        nodes,
        pred,
        nidx,
        is_full,
        mask_cpu,
        mask_gpu,
        js_,
        js,
        ∇,
        h∇,
        feature_names,
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
        target_mask_buf,
        tree_split_gpu,
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
        sums_temp_gpu, gains_per_feat_gpu,
        bins_per_feat_gpu,
        split_sums_temp_gpu,
        obliv_gains_gpu,
        obliv_count_gpu,
        group_cache
    )

    return m, cache
end
