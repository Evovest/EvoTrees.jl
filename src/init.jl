function init_core(params::EvoTypes, ::Type{CPU}, data, feature_names, y_train, w, offset)

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(data; feature_names, nbins=params.nbins, rng=params.rng)
    x_bin = binarize(data; feature_names, edges)
    nobs, nfeats = size(x_bin)

    T = Float32
    L = _loss2type_dict[params.loss]

    target_levels = nothing
    target_isordered = false
    if L == Logistic
        @assert eltype(y_train) <: Real && minimum(y_train) >= 0 && maximum(y_train) <= 1
        K = 1
        y = T.(y_train)
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == MLogLoss
        if eltype(y_train) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y_train)
            target_isordered = isordered(y_train)
            y = UInt32.(CategoricalArrays.levelcode.(y_train))
        elseif eltype(y_train) <: Integer || eltype(y_train) <: Bool || eltype(y_train) <: String || eltype(y_train) <: Char
            target_levels = sort(unique(y_train))
            yc = CategoricalVector(y_train, levels=target_levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        else
            @error "Invalid target eltype: $(eltype(y_train))"
        end
        K = length(target_levels)
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        @assert eltype(y_train) <: Real
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        @assert eltype(y_train) <: Real
        K = 2
        y = T.(y_train)
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = [mean(y)]
    end
    μ = T.(μ)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    @assert (length(y) == length(w) && minimum(w) > 0)

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

    # initialize model
    nodes = [TrainNode(nfeats, params.nbins, K, view(is, 1:0)) for n = 1:2^params.max_depth-1]
    bias = [Tree{L,K}(μ)]
    m = EvoTree{L,K}(L, K, bias, info)

    # build cache
    Y = typeof(y)
    N = typeof(first(nodes))
    cache = CacheBaseCPU{Y,N}(
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
        feature_names,
        featbins,
        feattypes,
        monotone_constraints,
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
    offset_name=nothing
)

    # set feature_names
    schema = Tables.schema(dtrain)
    _weight_name = isnothing(weight_name) ? Symbol("") : Symbol(weight_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_name = Symbol(target_name)
    if isnothing(feature_names)
        feature_names = Symbol[]
        for i in eachindex(schema.names)
            if schema.types[i] <: Union{Real,CategoricalValue}
                push!(feature_names, schema.names[i])
            end
        end
        feature_names = setdiff(feature_names, union([_target_name], [_weight_name], [_offset_name]))
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
    y_train = Tables.getcolumn(dtrain, _target_name)
    V = device_array_type(device)
    w = isnothing(weight_name) ? device_ones(device, T, nobs) : V{T}(Tables.getcolumn(dtrain, _weight_name))
    offset = isnothing(offset_name) ? nothing : V{T}(Tables.getcolumn(dtrain, _offset_name))

    m, cache = init_core(params, device, dtrain, feature_names, y_train, w, offset)

    return m, cache
end

# This should be different on CPUs and GPUs
device_ones(::Type{<:CPU}, ::Type{T}, n::Int) where {T} = ones(T, n)
device_array_type(::Type{<:CPU}) = Array

"""
    init(
        params::EvoTypes,
        x_train::AbstractMatrix,
        y_train::AbstractVector,
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
    y_train::AbstractVector,
    device::Type{<:Device}=CPU;
    feature_names=nothing,
    w_train=nothing,
    offset_train=nothing
)

    # initialize model and cache
    feature_names = isnothing(feature_names) ? [Symbol("feat_$i") for i in axes(x_train, 2)] : Symbol.(feature_names)
    @assert length(feature_names) == size(x_train, 2)

    T = Float32
    nobs = size(x_train, 1)
    V = device_array_type(device)
    w = isnothing(w_train) ? device_ones(device, T, nobs) : V{T}(w_train)
    offset = isnothing(offset_train) ? nothing : V{T}(offset_train)

    m, cache = init_core(params, device, x_train, feature_names, y_train, w, offset)

    return m, cache
end
