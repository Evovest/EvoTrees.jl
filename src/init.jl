"""
    orient_matrix_target(y, nobs)

Bring a matrix target to the internal layout `(n_targets, nobs)`.

The public matrix `fit` API takes `y` as `(nobs, n_targets)`, matching `x_train`
`(nobs, nfeats)`. Gradients are stored `(K, nobs)`, so this permutes once at the
boundary. A matrix that is already `(n_targets, nobs)` (second dim equals `nobs`,
first does not) is left as-is. Vectors are returned unchanged.
"""
function orient_matrix_target(y::AbstractVector, nobs::Integer)
    length(y) == nobs || error(
        "`y` has length $(length(y)) but there are $nobs observations. They must match."
    )
    return y
end
function orient_matrix_target(y::AbstractMatrix, nobs::Integer)
    nrows, ncols = size(y)
    if nrows == nobs
        return permutedims(y)
    elseif ncols == nobs
        return y
    else
        error(
            "`y` has size $(size(y)); one dimension must equal the number of observations ($nobs). " *
            "Pass `(nobs, n_targets)` to match `x_train`."
        )
    end
end

"""
    _init_target(::Type{L}, y_train, params, offset, ::Type{T})

Shared (device-agnostic) target/bias initialization: validates the target,
derives the output dimension `K`, the converted target `y` (host arrays;
device inits copy to their backend afterwards), and the initial bias `μ`.
Mutates `offset` in place into link space when provided. Single source of
truth for CPU (`src/init.jl`) and GPU (`ext/.../init.jl`) initialization.
"""
function _init_target(::Type{L}, y_train, params, offset, ::Type{T}) where {L,T}
    if (y_train isa AbstractMatrix) && !(L <: Union{GradientRegression, MLE2P, MAE, Quantile, Cred})
        error("Multi-target (matrix target) is supported for gradient-regression losses " *
              "(mse, logloss, poisson, gamma, tweedie), mae, quantile, the MLE losses " *
              "(gaussian_mle, logistic_mle), and credibility losses (cred_var, cred_std). " *
              "Got loss $(params.loss).")
    end
    target_levels = nothing
    target_isordered = false
    if L == LogLoss
        @assert eltype(y_train) <: Real && minimum(y_train) >= 0 && maximum(y_train) <= 1
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[logit(mean(y))]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[logit(mean(view(y, k, :))) for k in 1:K]
        end
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        @assert eltype(y_train) <: Real
        if L == Gamma
            ymin = minimum(y_train)
            ymin <= 0 && error(
                "Gamma regression requires a strictly positive target, got a minimum of $ymin. " *
                "The gamma deviance is undefined at 0."
            )
        end
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[log(mean(y))]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[log(mean(view(y, k, :))) for k in 1:K]
        end
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == MLogLoss
        if eltype(y_train) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y_train)
            target_isordered = isordered(y_train)
            y = UInt32.(CategoricalArrays.levelcode.(y_train))
        elseif eltype(y_train) <: Integer || eltype(y_train) <: Bool || eltype(y_train) <: String || eltype(y_train) <: Char
            yc = categorical(y_train, levels=sort(unique(y_train)), ordered=false)
            target_levels = CategoricalArrays.levels(yc)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        else
            error("Invalid target eltype: $(eltype(y_train))")
        end
        K = length(target_levels)
        K < 2 && error(
            "Classification requires a target with at least 2 levels, got $K: " *
            "$(string.(target_levels)). A single-class problem is not meaningful."
        )
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 2
            y = T.(y_train)
            μ = [mean(y), log(std(y))]
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        else
            Y = size(y_train, 1)
            K = 2 * Y
            y = T.(y_train)
            μ = T[]
            for t in 1:Y
                yt = view(y, t, :)
                push!(μ, mean(yt), log(std(yt)))
            end
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        end
    elseif L == LogisticMLE
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 2
            y = T.(y_train)
            μ = [mean(y), log(std(y) * sqrt(3) / π)]
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        else
            Y = size(y_train, 1)
            K = 2 * Y
            y = T.(y_train)
            μ = T[]
            for t in 1:Y
                yt = view(y, t, :)
                push!(μ, mean(yt), log(std(yt) * sqrt(3) / π))
            end
            !isnothing(offset) && unconstrain_mle_scale!(offset)
        end
    elseif L == MultiQuantile
        @assert eltype(y_train) <: Real
        K = length(params.alphas)
        y = T.(y_train)
        μ = T.(quantile.(Ref(y), params.alphas))
    elseif L <: Union{MAE,Quantile}
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[mean(y)]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[mean(view(y, k, :)) for k in 1:K]
        end
    elseif L <: Cred
        @assert eltype(y_train) <: Real
        if y_train isa AbstractVector
            K = 1
            y = T.(y_train)
            μ = T[mean(y)]
        else
            K = size(y_train, 1)
            y = T.(y_train)
            μ = T[mean(view(y, k, :)) for k in 1:K]
        end
    else
        @assert eltype(y_train) <: Real
        if L == LambdaRank
            # Scores are relative within a query, so a constant bias cancels.
            @assert minimum(y_train) >= 0 "`:lambdarank` requires non-negative graded relevance."
            K = 1
            y = T.(y_train)
            μ = T[0]
        elseif L <: GradientRegression
            if y_train isa AbstractVector
                K = 1
                y = T.(y_train)
                μ = T[mean(y)]
            else
                K = size(y_train, 1)
                y = T.(y_train)
                μ = T[mean(view(y, k, :)) for k in 1:K]
            end
        else
            K = 1
            y = T.(y_train)
            μ = [mean(y)]
        end
    end
    μ = T.(μ)
    return K, y, μ, target_levels, target_isordered
end

function init_core(params::EvoTypes, ::Type{CPU}, data, feature_names, y_train, w, offset, group=nothing)

    # binarize data into quantiles
    rng = Xoshiro(params.seed)

    edges, featbins, feattypes = get_edges(data; feature_names, nbins=params.nbins, rng)
    x_bin = binarize(data; feature_names, edges)
    x_bin_T = permutedims(x_bin)
    nobs, nfeats = size(x_bin)

    T = Float32
    L = _loss2type_dict[params.loss]

    K, y, μ, target_levels, target_isordered = _init_target(L, y_train, params, offset, T)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    @assert (size(y, ndims(y)) == length(w) && minimum(w) > 0)

    # initialize preds
    pred = zeros(T, K, nobs)
    pred .= μ
    !isnothing(offset) && (pred .+= offset')

    # initialize gradients
    ∇ = zeros(T, 2 * K + 1, nobs)
    ∇[end, :] .= w

    # initialize indexes
    mask_cond = zeros(UInt8, nobs)
    is = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)
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

    # Shared 4D hist storage (same layout as KA GPU cache). Each TrainNode
    # holds a contiguous view into trailing node dimension.
    nnodes = 2^params.max_depth - 1
    nbins = params.nbins
    h∇ = zeros(Float64, 2 * K + 1, nbins, nfeats, nnodes)
    nodes = [TrainNode(zero(Float64), view(is, 1:0), zeros(Float64, 2 * K + 1), zeros(Float64, 2 * K + 1), zeros(Float64, 2 * K + 1), view(h∇, :, :, :, n), zeros(nbins, nfeats)) for n = 1:nnodes]
    bias = [Tree{L,K}(μ)]
    m = EvoTree{L,K}(L, K, bias, info)

    # build cache
    Y = typeof(y)
    N = typeof(first(nodes))
    H = typeof(h∇)
    G = typeof(group)
    cache = CacheBaseCPU{Y,N,H,G}(
        rng,
        K,
        x_bin,
        x_bin_T,
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
        feature_names,
        featbins,
        feattypes,
        monotone_constraints,
        group,
    )
    return m, cache
end

"""
    init(
        params::EvoTypes,
        dtrain,
        device::Type{<:Device}=CPU;
        target_name,
        feature_names=nothing,
        weight_name=nothing,
        offset_name=nothing
    )

Initialise EvoTree
"""
function init(
    params::EvoTypes,
    dtrain,
    device::Type{<:Device}=CPU;
    target_name,
    feature_names=nothing,
    weight_name=nothing,
    offset_name=nothing,
    group_name=nothing
)

    # set feature_names
    schema = Tables.schema(dtrain)
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _group_name = isnothing(group_name) ? Symbol("") : Symbol(group_name)
    _target_names = target_name isa AbstractVector ? Symbol.(target_name) : [Symbol(target_name)]
    if isnothing(feature_names)
        feature_names = Symbol[]
        for i in eachindex(schema.names)
            if schema.types[i] <: Union{Real,CategoricalValue}
                push!(feature_names, schema.names[i])
            end
        end
        feature_names = setdiff(feature_names, union(_target_names, [_weight_name], [_offset_name], [_group_name]))
    else
        isa(feature_names, String) ? feature_names = [feature_names] : nothing
        feature_names = Symbol.(feature_names)
        @assert isa(feature_names, Vector{Symbol})
        @assert all(feature_names .∈ Ref(schema.names))
        for name in feature_names
            @assert schema.types[findfirst(name .== schema.names)] <: Union{Real,CategoricalValue}
        end
    end

    T = Float32
    nobs = length(Tables.getcolumn(dtrain, 1))
    y_train = length(_target_names) == 1 ?
        Tables.getcolumn(dtrain, _target_names[1]) :
        permutedims(reduce(hcat, [Tables.getcolumn(dtrain, t) for t in _target_names]))
    V = device_array_type(device)
    w = isnothing(weight_name) ? device_ones(device, T, nobs) : V{T}(Tables.getcolumn(dtrain, _weight_name))
    offset = isnothing(offset_name) ? nothing : V{T}(Tables.getcolumn(dtrain, _offset_name))
    group = isnothing(group_name) ? nothing : build_group_index(Tables.getcolumn(dtrain, _group_name), nobs, "group_name")

    m, cache = init_core(params, device, dtrain, feature_names, y_train, w, offset, group)

    m.info[:target_names] = _target_names
    m.info[:group_name] = isnothing(group_name) ? nothing : _group_name

    return m, cache
end

# This should be different on CPUs and GPUs
device_ones(::Type{<:CPU}, ::Type{T}, n::Int) where {T} = ones(T, n)
device_array_type(::Type{<:CPU}) = Array

"""
    init(
        params::EvoTypes,
        x_train::AbstractMatrix,
        y_train::AbstractVecOrMat,
        device::Type{<:Device}=CPU;
        feature_names=nothing,
        w_train=nothing,
        offset_train=nothing
    )

Initialise EvoTree
"""
function init(
    params::EvoTypes,
    x_train::AbstractMatrix,
    y_train::AbstractVecOrMat,
    device::Type{<:Device}=CPU;
    feature_names=nothing,
    w_train=nothing,
    offset_train=nothing,
    group_train=nothing
)

    # initialize model and cache
    feature_names = isnothing(feature_names) ? [Symbol("feat_$i") for i in axes(x_train, 2)] : Symbol.(feature_names)
    @assert length(feature_names) == size(x_train, 2)

    T = Float32
    nobs = size(x_train, 1)
    y_train = orient_matrix_target(y_train, nobs)
    V = device_array_type(device)
    w = isnothing(w_train) ? device_ones(device, T, nobs) : V{T}(w_train)
    offset = isnothing(offset_train) ? nothing : V{T}(offset_train)
    group = isnothing(group_train) ? nothing : build_group_index(group_train, nobs, "group_train")

    m, cache = init_core(params, device, x_train, feature_names, y_train, w, offset, group)

    return m, cache
end
