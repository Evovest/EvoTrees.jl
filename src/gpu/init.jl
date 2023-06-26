function init_core(params::EvoTypes{L}, ::Type{GPU}, data, fnames, y_train, w, offset) where {L}

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(data; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(binarize(data; fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32

    target_levels = nothing
    if L == Logistic
        K = 1
        y = T.(y_train)
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        K = 1
        y = T.(y_train)
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == MLogLoss
        if eltype(y_train) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y_train)
            y = UInt32.(CategoricalArrays.levelcode.(y_train))
        else
            target_levels = sort(unique(y_train))
            yc = CategoricalVector(y_train, levels=target_levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
        K = length(target_levels)
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        K = 1
        y = T.(y_train)
        μ = [mean(y)]
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
    h∇ = CUDA.zeros(Float64, 2 * K + 1, maximum(featbins), length(featbins))
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # initialize indexes
    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(eltype(js_), ceil(Int, params.colsample * nfeats))
    out = CUDA.zeros(UInt32, nobs)
    left = CUDA.zeros(UInt32, nobs)
    right = CUDA.zeros(UInt32, nobs)

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
    nodes = [TrainNode(featbins, K, view(is_in, 1:0)) for n = 1:2^params.max_depth-1]
    bias = [Tree{L,K}(μ)]
    m = EvoTree{L,K}(bias, info)

    # build cache
    cache = (
        info=Dict(:nrounds => 0),
        x_bin=x_bin,
        y=y,
        w=w,
        K=K,
        nodes=nodes,
        pred=pred,
        is_in=is_in,
        is_out=is_out,
        mask=mask,
        js_=js_,
        js=js,
        out=out,
        left=left,
        right=right,
        ∇=∇,
        h∇=h∇,
        fnames=fnames,
        edges=edges,
        featbins=featbins,
        feattypes=feattypes,
        feattypes_gpu=CuArray(feattypes),
        monotone_constraints=monotone_constraints,
    )
    return m, cache
end