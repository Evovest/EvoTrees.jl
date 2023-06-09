function init_target_pred(::EvoTypes{L,T}, y_raw, offset) where {L,T}

    target_levels = nothing
    if L == Logistic
        K = 1
        y = T.(y_raw)
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        K = 1
        y = T.(y_raw)
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == Softmax
        if eltype(y_raw) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y_raw)
            y = UInt32.(CategoricalArrays.levelcode.(y_raw))
        else
            target_levels = sort(unique(y_raw))
            yc = CategoricalVector(y_raw, levels=target_levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
        K = length(target_levels)
        μ = T.(log.(proportions(y_raw, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        K = 2
        y = T.(y_raw)
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        K = 2
        y = T.(y_raw)
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        K = 1
        y = T.(y_raw)
        μ = [mean(y)]
    end
    μ = T.(μ)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)

    return y, μ, K, target_levels

end

"""
    init_evotree(params::EvoTypes{T,U,S}, X::AbstractMatrix, Y::AbstractVector, W = nothing)
    
Initialise EvoTree
"""
function init(
    params::EvoTypes{L,T},
    dtrain::AbstractDataFrame;
    target_name,
    fnames=nothing,
    w_name=nothing,
    offset_name=nothing,
) where {L,T}

    # set fnames
    _w_name = isnothing(w_name) ? "" : string(w_name)
    _offset_name = isnothing(offset_name) ? "" : string(offset_name)
    if isnothing(fnames)
        fnames = String[]
        for name in names(dtrain)
            if eltype(dtrain[!, name]) <: Union{Real,CategoricalValue}
                push!(fnames, name)
            end
        end
        fnames = setdiff(fnames, union([target_name], [_w_name], [_offset_name]))
    else
        isa(fnames, String) ? fnames = [fnames] : nothing
        fnames = string.(fnames)
        @assert isa(fnames, Vector{String})
        for name in fnames
            @assert eltype(dtrain[!, name]) <: Union{Real,CategoricalValue}
        end
    end

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(dtrain; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = binarize(dtrain; fnames, edges)
    nobs, nfeats = size(x_bin)

    w = isnothing(w_name) ? ones(T, nobs) : Vector{T}(dtrain[!, w_name])
    offset = !isnothing(offset_name) ? T.(dtrain[!, offset_name]) : nothing
    y_raw = dtrain[!, target_name]

    y, μ, K, target_levels = init_target_pred(params, y_raw, offset)
    # m, cache = init(params, data, fnames, y_raw, w, offset)

    @assert (length(y) == length(w) && minimum(w) > 0)
    # initialize gradients
    ∇ = zeros(T, 2 * K + 1, nobs)
    ∇[end, :] .= w

    # initialize preds
    pred = zeros(T, K, nobs)
    pred .= μ
    !isnothing(offset) && (pred .+= offset')

    # init EvoTree
    bias = [Tree{L,K,T}(μ)]

    is_in = zeros(UInt32, nobs)
    is_out = zeros(UInt32, nobs)
    mask = zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))

    # initialize histograms
    nodes = [TrainNode(featbins, K, view(is_in, 1:0), T) for n = 1:2^params.max_depth-1]
    out = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :fnames => fnames,
        :target_name => target_name,
        :w_name => w_name,
        :offset_name => offset_name,
        :target_levels => target_levels,
        :edges => edges,
        :feattypes => feattypes,
    )

    # initialize model
    m = EvoTree{L,K,T}(bias, info)

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
