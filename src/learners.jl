# make a Random Number Generator object
mk_rng(rng::AbstractRNG) = rng
mk_rng(int::Integer) = Random.MersenneTwister(int)

mutable struct EvoTreeRegressor <: MMI.Deterministic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    bagging_size::Int
    early_stopping_rounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    nbins::Int
    alpha::Float64
    monotone_constraints::Dict{Int,Int}
    tree_type::Symbol
    rng::AbstractRNG
    device::Symbol
end

function EvoTreeRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :metric => nothing,
        :nrounds => 100,
        :bagging_size => 1,
        :early_stopping_rounds => typemax(Int),
        :L2 => 1.0,
        :lambda => 0.0,
        :gamma => 0.0,
        :eta => 0.1,
        :max_depth => 6,
        :min_weight => 1.0,
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => :binary,
        :rng => 123,
        :device => :cpu
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    _loss_list = [:mse, :logloss, :poisson, :gamma, :tweedie, :mae, :quantile, :cred_std, :cred_var]
    loss = Symbol(args[:loss])
    if loss == :linear
        loss = :mse
        @warn "`:linear` loss is no longer supported - `:mse` loss will be used."
    end
    if loss == :logistic
        loss = :logloss
        @warn "`:logistic` loss is no longer supported - `:logloss` will be used."
    end
    if loss ∉ _loss_list
        error("Invalid loss. Must be one of: $_loss_list")
    end

    _metric_list = [:mse, :rmse, :mae, :logloss, :poisson, :gamma, :tweedie, :quantile, :gini]
    if isnothing(args[:metric])
        if loss ∈ [:cred_std, :cred_var]
            metric = :mae
        else
            metric = loss
        end
    else
        metric = Symbol(args[:metric])
    end
    if metric ∉ _metric_list
        error("Invalid metric. Must be one of: $_metric_list")
    end

    tree_type = Symbol(args[:tree_type])
    device = Symbol(args[:device])
    rng = mk_rng(args[:rng])
    check_args(args)

    model = EvoTreeRegressor(
        loss,
        metric,
        args[:nrounds],
        args[:bagging_size],
        args[:early_stopping_rounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        args[:alpha],
        args[:monotone_constraints],
        tree_type,
        rng,
        device
    )

    return model
end

mutable struct EvoTreeCount <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    bagging_size::Int
    early_stopping_rounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    nbins::Int
    monotone_constraints::Dict{Int,Int}
    tree_type::Symbol
    rng::AbstractRNG
    device::Symbol
end

function EvoTreeCount(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :bagging_size => 1,
        :early_stopping_rounds => typemax(Int),
        :L2 => 1.0,
        :lambda => 0.0,
        :gamma => 0.0,
        :eta => 0.1,
        :max_depth => 6,
        :min_weight => 1.0,
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => :binary,
        :rng => 123,
        :device => :cpu
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    loss = :poisson
    metric = :poisson

    tree_type = Symbol(args[:tree_type])
    device = Symbol(args[:device])
    rng = mk_rng(args[:rng])
    check_args(args)

    model = EvoTreeCount(
        loss,
        metric,
        args[:nrounds],
        args[:bagging_size],
        args[:early_stopping_rounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        args[:monotone_constraints],
        tree_type,
        rng,
        device
    )

    return model
end

mutable struct EvoTreeClassifier <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    bagging_size::Int
    early_stopping_rounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    nbins::Int
    tree_type::Symbol
    rng::AbstractRNG
    device::Symbol
end

function EvoTreeClassifier(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :bagging_size => 1,
        :early_stopping_rounds => typemax(Int),
        :L2 => 1.0,
        :lambda => 0.0,
        :gamma => 0.0,
        :eta => 0.1,
        :max_depth => 6,
        :min_weight => 1.0,
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :tree_type => :binary,
        :rng => 123,
        :device => :cpu
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    loss = :mlogloss
    metric = :mlogloss

    tree_type = Symbol(args[:tree_type])
    device = Symbol(args[:device])
    rng = mk_rng(args[:rng])
    check_args(args)

    model = EvoTreeClassifier(
        loss,
        metric,
        args[:nrounds],
        args[:bagging_size],
        args[:early_stopping_rounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        tree_type,
        rng,
        device
    )

    return model
end

mutable struct EvoTreeMLE <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    bagging_size::Int
    early_stopping_rounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    nbins::Int
    monotone_constraints::Dict{Int,Int}
    tree_type::Symbol
    rng::AbstractRNG
    device::Symbol
end

function EvoTreeMLE(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :gaussian_mle,
        :metric => nothing,
        :nrounds => 100,
        :bagging_size => 1,
        :early_stopping_rounds => typemax(Int),
        :L2 => 1.0,
        :lambda => 0.0,
        :gamma => 0.0,
        :eta => 0.1,
        :max_depth => 6,
        :min_weight => 8.0,
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => :binary,
        :rng => 123,
        :device => :cpu
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    _loss_list = [:gaussian_mle, :logistic_mle]
    loss = Symbol(args[:loss])
    if loss ∉ _loss_list
        error("Invalid loss. Must be one of: $_loss_list")
    end

    _metric_list = [:gaussian_mle, :logistic_mle]
    if isnothing(args[:metric])
        metric = loss
    end
    if metric ∉ _metric_list
        error("Invalid metric. Must be one of: $_metric_list")
    end

    tree_type = Symbol(args[:tree_type])
    device = Symbol(args[:device])
    rng = mk_rng(args[:rng])
    check_args(args)

    model = EvoTreeMLE(
        loss,
        metric,
        args[:nrounds],
        args[:bagging_size],
        args[:early_stopping_rounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        args[:monotone_constraints],
        tree_type,
        rng,
        device
    )

    return model
end

mutable struct EvoTreeGaussian <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    bagging_size::Int
    early_stopping_rounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    nbins::Int
    monotone_constraints::Dict{Int,Int}
    tree_type::Symbol
    rng::AbstractRNG
    device::Symbol
end
function EvoTreeGaussian(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :bagging_size => 1,
        :early_stopping_rounds => typemax(Int),
        :L2 => 1.0,
        :lambda => 0.0,
        :gamma => 0.0,
        :eta => 0.1,
        :max_depth => 6,
        :min_weight => 8.0,
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => :binary,
        :rng => 123,
        :device => :cpu
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    loss = :gaussian_mle
    metric = :gaussian_mle

    tree_type = Symbol(args[:tree_type])
    device = Symbol(args[:device])
    rng = mk_rng(args[:rng])
    check_args(args)

    model = EvoTreeGaussian(
        loss,
        metric,
        args[:nrounds],
        args[:bagging_size],
        args[:early_stopping_rounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        args[:monotone_constraints],
        tree_type,
        rng,
        device
    )

    return model
end

const EvoTypes = Union{
    EvoTreeRegressor,
    EvoTreeCount,
    EvoTreeClassifier,
    EvoTreeGaussian,
    EvoTreeMLE,
}

function Base.show(io::IO, config::EvoTypes)
    println(io, "$(typeof(config))")
    for fname in fieldnames(typeof(config))
        println(io, " - $fname: $(getfield(config, fname))")
    end
end

"""
    check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}

Check model parameter if it's valid
"""
function check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}
    min_value = max(typemin(T), min_value)
    max_value = min(typemax(T), max_value)
    try
        convert(T, value)
        @assert min_value <= value <= max_value
    catch
        error("Invalid value for parameter `$(string(label))`: $value. `$(string(label))` must be of type $T with value between $min_value and $max_value.")
    end
end

"""
    check_args(args::Dict{Symbol,Any})

Check model arguments if they are valid
"""
function check_args(args::Dict{Symbol,Any})

    # Check integer parameters
    check_parameter(Int, args[:nrounds], 0, typemax(Int), :nrounds)
    check_parameter(Int, args[:max_depth], 1, typemax(Int), :max_depth)
    check_parameter(Int, args[:nbins], 2, 255, :nbins)

    # check positive float parameters
    check_parameter(Float64, args[:lambda], zero(Float64), typemax(Float64), :lambda)
    check_parameter(Float64, args[:gamma], zero(Float64), typemax(Float64), :gamma)
    check_parameter(Float64, args[:min_weight], zero(Float64), typemax(Float64), :min_weight)

    # check bounded parameters
    check_parameter(Float64, args[:rowsample], eps(Float64), one(Float64), :rowsample)
    check_parameter(Float64, args[:colsample], eps(Float64), one(Float64), :colsample)
    check_parameter(Float64, args[:eta], zero(Float64), typemax(Float64), :eta)

    try
        tree_type = string(args[:tree_type])
        @assert tree_type ∈ ["binary", "oblivious"]
    catch
        error("Invalid input for `tree_type` parameter: `$(args[:tree_type])`. Must be of one of `binary` or `oblivious`")
    end

end

"""
    check_args(model::EvoTypes)

Check model arguments if they are valid (eg, after mutation when tuning hyperparams)
Note: does not check consistency of model type and loss selected
"""
function check_args(model::EvoTypes)

    # Check integer parameters
    check_parameter(Int, model.max_depth, 1, typemax(Int), :max_depth)
    check_parameter(Int, model.nrounds, 0, typemax(Int), :nrounds)
    check_parameter(Int, model.nbins, 2, 255, :nbins)

    # check positive float parameters
    check_parameter(Float64, model.lambda, zero(Float64), typemax(Float64), :lambda)
    check_parameter(Float64, model.gamma, zero(Float64), typemax(Float64), :gamma)
    check_parameter(Float64, model.min_weight, zero(Float64), typemax(Float64), :min_weight)

    # check bounded parameters
    check_parameter(Float64, model.rowsample, eps(Float64), one(Float64), :rowsample)
    check_parameter(Float64, model.colsample, eps(Float64), one(Float64), :colsample)
    check_parameter(Float64, model.eta, zero(Float64), typemax(Float64), :eta)

    try
        tree_type = string(model.tree_type)
        @assert tree_type ∈ ["binary", "oblivious"]
    catch
        error("Invalid input for `tree_type` parameter: `$(model.tree_type)`. Must be of one of `binary` or `oblivious`")
    end
    return nothing
end
