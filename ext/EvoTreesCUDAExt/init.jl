function EvoTrees.init_core(params::EvoTrees.EvoTypes, ::Type{<:EvoTrees.GPU}, data, feature_names, y_train, w, offset)

    # binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(data; feature_names, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(EvoTrees.binarize(data; feature_names, edges))
    nobs, nfeats = size(x_bin)

    T = Float32
    L = EvoTrees._loss2type_dict[params.loss]

    target_levels = nothing
    target_isordered = false
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
    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)

    # initialize preds
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # initialize gradients
    h∇_cpu = zeros(Float32, 2 * K + 1, maximum(featbins), length(featbins))
    h∇ = CuArray(h∇_cpu)
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # initialize indexes
    mask_cond = CUDA.zeros(UInt8, nobs)
    is = CUDA.zeros(UInt32, nobs)
    left = CUDA.zeros(UInt32, nobs)
    right = CUDA.zeros(UInt32, nobs)
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    # initialize model
    nodes = [EvoTrees.TrainNode(nfeats, params.nbins, K, view(is, 1:0)) for n = 1:2^params.max_depth-1]
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(L, K, bias, info)

    # build cache
    Y = typeof(y)
    N = typeof(nodes)
    feattypes_gpu = CuArray(feattypes)
    cache = CacheBaseGPU{Y,N}(
        K,
        x_bin,
        y,
        w,
        pred,
        nodes,
        mask_cond,
        is,
        left,
        right,
        js,
        ∇,
        h∇,
        h∇_cpu,
        feature_names,
        featbins,
        feattypes,
        feattypes_gpu,
        monotone_constraints,
    )
    return m, cache
end
