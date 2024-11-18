abstract type ModelType end
abstract type GradientRegression <: ModelType end
abstract type MLE2P <: ModelType end # 2-parameters max-likelihood

abstract type MSE <: GradientRegression end
abstract type LogLoss <: GradientRegression end
abstract type Poisson <: GradientRegression end
abstract type Gamma <: GradientRegression end
abstract type Tweedie <: GradientRegression end
abstract type MLogLoss <: ModelType end
abstract type GaussianMLE <: MLE2P end
abstract type LogisticMLE <: MLE2P end
abstract type Quantile <: ModelType end
abstract type L1 <: ModelType end
abstract type Cred <: ModelType end
abstract type CredV1A <: Cred end
abstract type CredV2A <: Cred end
abstract type CredV1B <: Cred end
abstract type CredV2B <: Cred end

# Converts MSE -> :mse
const _type2loss_dict = Dict(
    MSE => :mse,
    LogLoss => :logloss,
    Poisson => :poisson,
    Gamma => :gamma,
    Tweedie => :tweedie,
    MLogLoss => :mlogloss,
    GaussianMLE => :gaussian_mle,
    LogisticMLE => :logistic_mle,
    Quantile => :quantile,
    L1 => :l1,
    CredV1A => :credV1A,
    CredV2A => :credV2A,
    CredV1B => :credV1B,
    CredV2B => :credV2B,
)
_type2loss(L::Type) = _type2loss_dict[L]

# make a Random Number Generator object
mk_rng(rng::AbstractRNG) = rng
mk_rng(int::Integer) = Random.MersenneTwister(int)

mutable struct EvoTreeRegressor{L<:ModelType} <: MMI.Deterministic
    nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    nbins::Int
    alpha::Float64
    monotone_constraints::Any
    tree_type::String
    rng::Any
end

function EvoTreeRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => "binary",
        :rng => 123,
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    args[:loss] = Symbol(args[:loss])

    if args[:loss] == :mse
        L = MSE
    elseif args[:loss] == :linear
        L = MSE
    elseif args[:loss] == :logloss
        L = LogLoss
    elseif args[:loss] == :logistic
        L = LogLoss
    elseif args[:loss] == :gamma
        L = Gamma
    elseif args[:loss] == :tweedie
        L = Tweedie
    elseif args[:loss] == :l1
        L = L1
    elseif args[:loss] == :quantile
        L = Quantile
    elseif args[:loss] == :credS1A
        L = CredS1A
    elseif args[:loss] == :credS1B
        L = CredS1B
    elseif args[:loss] == :credV1A
        L = CredV1A
    elseif args[:loss] == :credV2A
        L = CredV2A
    elseif args[:loss] == :credV1B
        L = CredV1B
    elseif args[:loss] == :credV2B
        L = CredV2B
    else
        error(
            "Invalid loss: $(args[:loss]). Only [`:mse`, `:logloss`, `:gamma`, `:tweedie`, `:l1`, `:quantile`] are supported by EvoTreeRegressor.",
        )
    end

    check_args(args)

    model = EvoTreeRegressor{L}(
        args[:nrounds],
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
        args[:tree_type],
        args[:rng],
    )

    return model
end

function EvoTreeRegressor{L}(; kwargs...) where {L}
    EvoTreeRegressor(; loss=_type2loss(L), kwargs...)
end

mutable struct EvoTreeCount{L<:ModelType} <: MMI.Probabilistic
    nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    nbins::Int
    alpha::Float64
    monotone_constraints::Any
    tree_type::String
    rng::Any
end

function EvoTreeCount(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => "binary",
        :rng => 123,
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    L = Poisson
    check_args(args)

    model = EvoTreeCount{L}(
        args[:nrounds],
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
        args[:tree_type],
        args[:rng],
    )

    return model
end

function EvoTreeCount{L}(; kwargs...) where {L}
    EvoTreeCount(; kwargs...)
end

mutable struct EvoTreeClassifier{L<:ModelType} <: MMI.Probabilistic
    nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    nbins::Int
    alpha::Float64
    tree_type::String
    rng::Any
end

function EvoTreeClassifier(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :tree_type => "binary",
        :rng => 123,
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    L = MLogLoss
    check_args(args)

    model = EvoTreeClassifier{L}(
        args[:nrounds],
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
        args[:tree_type],
        args[:rng],
    )

    return model
end

function EvoTreeClassifier{L}(; kwargs...) where {L}
    EvoTreeClassifier(; kwargs...)
end

mutable struct EvoTreeMLE{L<:ModelType} <: MMI.Probabilistic
    nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    nbins::Int
    alpha::Float64
    monotone_constraints::Any
    tree_type::String
    rng::Any
end

function EvoTreeMLE(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :gaussian_mle,
        :nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 8.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => "binary",
        :rng => 123,
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    args[:loss] = Symbol(args[:loss])

    if args[:loss] in [:gaussian, :gaussian_mle]
        L = GaussianMLE
    elseif args[:loss] in [:logistic, :logistic_mle]
        L = LogisticMLE
    else
        error(
            "Invalid loss: $(args[:loss]). Only `:gaussian_mle` and `:logistic_mle` are supported by EvoTreeMLE.",
        )
    end

    check_args(args)

    model = EvoTreeMLE{L}(
        args[:nrounds],
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
        args[:tree_type],
        args[:rng],
    )

    return model
end

function EvoTreeMLE{L}(; kwargs...) where {L}
    if L == GaussianMLE
        loss = :gaussian_mle
    elseif L == LogisticMLE
        loss = :logistic_mle
    end
    EvoTreeMLE(; loss=loss, kwargs...)
end


mutable struct EvoTreeGaussian{L<:ModelType} <: MMI.Probabilistic
    nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    nbins::Int
    alpha::Float64
    monotone_constraints::Any
    tree_type::String
    rng::Any
end
function EvoTreeGaussian(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 8.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => "binary",
        :rng => 123,
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    L = GaussianMLE
    check_args(args)

    model = EvoTreeGaussian{L}(
        args[:nrounds],
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
        args[:tree_type],
        args[:rng],
    )

    return model
end

function EvoTreeGaussian{L}(; kwargs...) where {L}
    EvoTreeGaussian(; kwargs...)
end

const EvoTypes{L} = Union{
    EvoTreeRegressor{L},
    EvoTreeCount{L},
    EvoTreeClassifier{L},
    EvoTreeGaussian{L},
    EvoTreeMLE{L},
}

_get_struct_loss(::EvoTypes{L}) where {L} = L

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
    check_parameter(Float64, args[:alpha], zero(Float64), one(Float64), :alpha)
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
    check_args(model::EvoTypes{L}) where {L}

Check model arguments if they are valid (eg, after mutation when tuning hyperparams)
Note: does not check consistency of model type and loss selected
"""
function check_args(model::EvoTypes{L}) where {L}

    # Check integer parameters
    check_parameter(Int, model.max_depth, 1, typemax(Int), :max_depth)
    check_parameter(Int, model.nrounds, 0, typemax(Int), :nrounds)
    check_parameter(Int, model.nbins, 2, 255, :nbins)

    # check positive float parameters
    check_parameter(Float64, model.lambda, zero(Float64), typemax(Float64), :lambda)
    check_parameter(Float64, model.gamma, zero(Float64), typemax(Float64), :gamma)
    check_parameter(Float64, model.min_weight, zero(Float64), typemax(Float64), :min_weight)

    # check bounded parameters
    check_parameter(Float64, model.alpha, zero(Float64), one(Float64), :alpha)
    check_parameter(Float64, model.rowsample, eps(Float64), one(Float64), :rowsample)
    check_parameter(Float64, model.colsample, eps(Float64), one(Float64), :colsample)
    check_parameter(Float64, model.eta, zero(Float64), typemax(Float64), :eta)

    try
        tree_type = string(model.tree_type)
        @assert tree_type ∈ ["binary", "oblivious"]
    catch
        error("Invalid input for `tree_type` parameter: `$(model.tree_type)`. Must be of one of `binary` or `oblivious`")
    end
end