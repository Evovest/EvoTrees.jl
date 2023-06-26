function init_core(params::EvoTypes{L}, ::Type{CPU}, data, fnames, y_train, w, offset) where {L}

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(data; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = binarize(data; fnames, edges)
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
    is_in = zeros(UInt32, nobs)
    is_out = zeros(UInt32, nobs)
    mask = zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))
    out = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)

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
        pred=pred,
        K=K,
        nodes=nodes,
        is_in=is_in,
        is_out=is_out,
        mask=mask,
        js_=js_,
        js=js,
        out=out,
        left=left,
        right=right,
        ∇=∇,
        edges=edges,
        fnames=fnames,
        featbins=featbins,
        feattypes=feattypes,
        monotone_constraints=monotone_constraints,
    )
    return m, cache
end

"""
    init(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)

Initialise EvoTree
"""
function init(
    params::EvoTypes,
    dtrain,
    device::Type{<:Device}=CPU;
    target_name,
    fnames=nothing,
    w_name=nothing,
    offset_name=nothing
)

    # set fnames
    schema = Tables.schema(dtrain)
    _w_name = isnothing(w_name) ? Symbol("") : Symbol(w_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_name = Symbol(target_name)
    if isnothing(fnames)
        fnames = Symbol[]
        for i in eachindex(schema.names)
            if schema.types[i] <: Union{Real,CategoricalValue}
                push!(fnames, schema.names[i])
            end
        end
        fnames = setdiff(fnames, union([_target_name], [_w_name], [_offset_name]))
    else
        isa(fnames, String) ? fnames = [fnames] : nothing
        fnames = Symbol.(fnames)
        @assert isa(fnames, Vector{Symbol})
        @assert all(fnames .∈ Ref(schema.names))
        for name in fnames
            @assert schema.types[findfirst(name .== schema.names)] <: Union{Real,CategoricalValue}
        end
    end

    T = Float32
    nobs = length(Tables.getcolumn(dtrain, 1))
    y_train = Tables.getcolumn(dtrain, _target_name)
    if device <: GPU
        w = isnothing(w_name) ? CUDA.ones(T, nobs) : CuArray{T}(Tables.getcolumn(dtrain, _w_name))
        offset = !isnothing(offset_name) ? CuArray{T}(Tables.getcolumn(dtrain, _offset_name)) : nothing
    else
        w = isnothing(w_name) ? ones(T, nobs) : Vector{T}(Tables.getcolumn(dtrain, _w_name))
        offset = !isnothing(offset_name) ? T.(Tables.getcolumn(dtrain, _offset_name)) : nothing
    end

    m, cache = init_core(params, device, dtrain, fnames, y_train, w, offset)

    return m, cache
end


"""
    init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)

Initialise EvoTree
"""
function init(
    params::EvoTypes,
    x_train::AbstractMatrix,
    y_train::AbstractVector,
    device::Type{<:Device}=CPU;
    fnames=nothing,
    w_train=nothing,
    offset_train=nothing
)

    # initialize model and cache
    fnames = isnothing(fnames) ? [Symbol("feat_$i") for i in axes(x_train, 2)] : Symbol.(fnames)
    @assert length(fnames) == size(x_train, 2)

    T = Float32
    nobs = size(x_train, 1)
    if device <: GPU
        w = isnothing(w_train) ? CUDA.ones(T, nobs) : CuArray{T}(w_train)
        offset = !isnothing(offset_train) ? CuArray{T}(offset_train) : nothing
    else
        w = isnothing(w_train) ? ones(T, nobs) : Vector{T}(w_train)
        offset = !isnothing(offset_train) ? T.(offset_train) : nothing
    end

    m, cache = init_core(params, device, x_train, fnames, y_train, w, offset)

    return m, cache
end
