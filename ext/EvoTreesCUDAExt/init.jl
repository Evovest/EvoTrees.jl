function EvoTrees.init_core(params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}, data, fnames, y_train, w, offset) where {L}

    # binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(data; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(EvoTrees.binarize(data; fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32

    target_levels = nothing
    if L == EvoTrees.Logistic
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
    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)

    # initialize preds
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # initialize gradients
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    h∇ = CUDA.zeros(Float32, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    h∇L = CUDA.zero(h∇)
    h∇R = CUDA.zero(h∇)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # initialize indexes
    nidx = CUDA.ones(UInt32, nobs)
    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(eltype(js_), ceil(Int, params.colsample * nfeats))

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    info = Dict(
        :fnames => fnames,
        :target_levels => target_levels,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    # initialize model
    nodes = [EvoTrees.TrainNode(featbins, K) for _ in 1:2^params.max_depth-1]
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(bias, info)

    cond_feats = zeros(Int, 2^(params.max_depth - 1) - 1)
    cond_bins = zeros(UInt8, 2^(params.max_depth - 1) - 1)
    cond_feats_gpu = CuArray(cond_feats)
    cond_bins_gpu = CuArray(cond_bins)
    feattypes_gpu = CuArray(feattypes)
    monotone_constraints_gpu = CuArray(monotone_constraints)

    # preallocate buffers used across depths
    max_nodes_level = 2^params.max_depth
    left_nodes_buf = CUDA.zeros(Int32, max_nodes_level)
    right_nodes_buf = CUDA.zeros(Int32, max_nodes_level)
    target_mask_buf = CUDA.zeros(UInt8, 2^(params.max_depth + 1))

    cache = (
        info=Dict(:nrounds => 0),
        x_bin=x_bin,
        y=y,
        w=w,
        K=K,
        nodes=nodes,
        pred=pred,
        nidx=nidx,
        is_in=is_in,
        is_out=is_out,
        mask=mask,
        js_=js_,
        js=js,
        ∇=∇,
        h∇=h∇,
        h∇L=h∇L,
        h∇R=h∇R,
        fnames=fnames,
        edges=edges,
        featbins=featbins,
        feattypes_gpu=feattypes_gpu,
        cond_feats=cond_feats,
        cond_feats_gpu=cond_feats_gpu,
        cond_bins=cond_bins,
        cond_bins_gpu=cond_bins_gpu,
        monotone_constraints_gpu=monotone_constraints_gpu,
        left_nodes_buf=left_nodes_buf,
        right_nodes_buf=right_nodes_buf,
        target_mask_buf=target_mask_buf,
    )
    return m, cache
end

