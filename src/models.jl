abstract type ModelType end
abstract type GradientRegression <: ModelType end
abstract type L1Regression <: ModelType end
abstract type QuantileRegression <: ModelType end
abstract type MultiClassRegression <: ModelType end
abstract type MLE2P <: ModelType end # 2-parameters max-likelihood
struct Linear <: GradientRegression end
struct Logistic <: GradientRegression end
struct Poisson <: GradientRegression end
struct Gamma <: GradientRegression end
struct Tweedie <: GradientRegression end
struct L1 <: L1Regression end
struct Quantile <: QuantileRegression end
struct Softmax <: MultiClassRegression end
struct GaussianMLE <: MLE2P end
struct LogisticMLE <: MLE2P end

# make a Random Number Generator object
mk_rng(rng::AbstractRNG, device = "cpu") = rng
function mk_rng(int::Integer, device = "cpu")
    if VERSION < v"1.7"
        rng = Random.MersenneTwister()
    else
        rng = Random.TaskLocalRNG()
    end
    seed!(rng, int)
    device == "gpu" && CUDA.seed!(int)
    return rng
end

mutable struct EvoTreeRegressor{L<:ModelType,T} <: MMI.Deterministic
    nrounds::Int
    lambda::T
    gamma::T
    eta::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::Int
    alpha::T
    monotone_constraints::Any
    rng::Any
    device::Any
end

function EvoTreeRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float32,
        :loss => :linear,
        :nrounds => 10,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 5,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 32,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :rng => 123,
        :device => "cpu",
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng], String(args[:device]))
    args[:loss] = Symbol(args[:loss])
    T = args[:T]

    if args[:loss] == :linear
        L = Linear
    elseif args[:loss] == :logistic
        L = Logistic
    elseif args[:loss] == :gamma
        L = Gamma
    elseif args[:loss] == :tweedie
        L = Tweedie
    elseif args[:loss] == :L1
        L = L1
    elseif args[:loss] == :quantile
        L = Quantile
    else
        error(
            "Invalid loss: $(args[:loss]). Only [`:linear`, `:logistic`, `:L1`, `:quantile`] are supported at the moment by EvoTreeRegressor.",
        )
    end

    model = EvoTreeRegressor{L,T}(
        args[:nrounds],
        T(args[:lambda]),
        T(args[:gamma]),
        T(args[:eta]),
        args[:max_depth],
        T(args[:min_weight]),
        T(args[:rowsample]),
        T(args[:colsample]),
        args[:nbins],
        T(args[:alpha]),
        args[:monotone_constraints],
        args[:rng],
        args[:device],
    )

    return model
end


mutable struct EvoTreeCount{L<:ModelType,T} <: MMI.Probabilistic
    nrounds::Int
    lambda::T
    gamma::T
    eta::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::Int
    alpha::T
    monotone_constraints::Any
    rng::Any
    device::Any
end

function EvoTreeCount(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float32,
        :nrounds => 10,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 5,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 32,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :rng => 123,
        :device => "cpu",
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng], String(args[:device]))
    L = Poisson
    T = args[:T]

    model = EvoTreeCount{L,T}(
        args[:nrounds],
        T(args[:lambda]),
        T(args[:gamma]),
        T(args[:eta]),
        args[:max_depth],
        T(args[:min_weight]),
        T(args[:rowsample]),
        T(args[:colsample]),
        args[:nbins],
        T(args[:alpha]),
        args[:monotone_constraints],
        args[:rng],
        args[:device],
    )

    return model
end

mutable struct EvoTreeClassifier{L<:ModelType,T} <: MMI.Probabilistic
    nrounds::Int
    lambda::T
    gamma::T
    eta::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::Int
    alpha::T
    rng::Any
    device::Any
end

function EvoTreeClassifier(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float32,
        :nrounds => 10,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 5,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 32,
        :alpha => 0.5,
        :rng => 123,
        :device => "cpu",
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng], String(args[:device]))
    L = Softmax
    T = args[:T]

    model = EvoTreeClassifier{L,T}(
        args[:nrounds],
        T(args[:lambda]),
        T(args[:gamma]),
        T(args[:eta]),
        args[:max_depth],
        T(args[:min_weight]),
        T(args[:rowsample]),
        T(args[:colsample]),
        args[:nbins],
        T(args[:alpha]),
        args[:rng],
        args[:device],
    )

    return model
end

mutable struct EvoTreeMLE{L<:ModelType,T} <: MMI.Probabilistic
    nrounds::Int
    lambda::T
    gamma::T
    eta::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::Int
    alpha::T
    monotone_constraints::Any
    rng::Any
    device::Any
end

function EvoTreeMLE(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float64,
        :loss => :gaussian,
        :nrounds => 10,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 5,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 32,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :rng => 123,
        :device => "cpu",
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng], String(args[:device]))
    args[:loss] = Symbol(args[:loss])
    T = args[:T]

    if args[:loss] in [:gaussian, :gaussian_mle]
        L = GaussianMLE
    elseif args[:loss] in [:logistic, :logistic_mle]
        L = LogisticMLE
    else
        error(
            "Invalid loss: $(args[:loss]). Only `:gaussian` / `:gaussian_mle` and `:logistic` / `:logistic_mle` are supported at the moment by EvoTreeMLE.",
        )
    end

    model = EvoTreeMLE{L,T}(
        args[:nrounds],
        T(args[:lambda]),
        T(args[:gamma]),
        T(args[:eta]),
        args[:max_depth],
        T(args[:min_weight]),
        T(args[:rowsample]),
        T(args[:colsample]),
        args[:nbins],
        T(args[:alpha]),
        args[:monotone_constraints],
        args[:rng],
        args[:device],
    )

    return model
end


mutable struct EvoTreeGaussian{L<:ModelType,T} <: MMI.Probabilistic
    nrounds::Int
    lambda::T
    gamma::T
    eta::T
    max_depth::Int
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::Int
    alpha::T
    monotone_constraints::Any
    rng::Any
    device::Any
end
function EvoTreeGaussian(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float64,
        :nrounds => 10,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 5,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :nbins => 32,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :rng => 123,
        :device => "cpu",
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng], String(args[:device]))
    L = GaussianMLE
    T = args[:T]

    model = EvoTreeGaussian{L,T}(
        args[:nrounds],
        T(args[:lambda]),
        T(args[:gamma]),
        T(args[:eta]),
        args[:max_depth],
        T(args[:min_weight]),
        T(args[:rowsample]),
        T(args[:colsample]),
        args[:nbins],
        T(args[:alpha]),
        args[:monotone_constraints],
        args[:rng],
        args[:device],
    )

    return model
end

const EvoTypes{L,T} = Union{
    EvoTreeRegressor{L,T},
    EvoTreeCount{L,T},
    EvoTreeClassifier{L,T},
    EvoTreeGaussian{L,T},
    EvoTreeMLE{L,T},
}

get_types(::EvoTypes{L,T}) where {L,T} = (L, T)

function Base.show(io::IO, config::EvoTypes)
    println(io, "$(typeof(config))")
    for fname in fieldnames(typeof(config))
        println(io, " - $fname: $(getfield(config, fname))")
    end
end