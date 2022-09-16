abstract type ModelType end
abstract type GradientRegression <: ModelType end
abstract type L1Regression <: ModelType end
abstract type QuantileRegression <: ModelType end
abstract type MultiClassRegression <: ModelType end
abstract type GaussianRegression <: ModelType end
struct Linear <: GradientRegression end
struct Logistic <: GradientRegression end
struct Poisson <: GradientRegression end
struct Gamma <: GradientRegression end
struct Tweedie <: GradientRegression end
struct L1 <: L1Regression end
struct Quantile <: QuantileRegression end
struct Softmax <: MultiClassRegression end
struct Gaussian <: GaussianRegression end

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

mutable struct EvoTreeRegressor{T<:AbstractFloat,U<:ModelType,S<:Int} <: MMI.Deterministic
    loss::U
    nrounds::S
    lambda::T
    gamma::T
    eta::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    alpha::T
    monotone_constraints
    metric::Symbol
    rng
    device
end

function EvoTreeRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float64,
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
        :metric => :mse,
        :rng => 123,
        :device => "cpu"
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 && @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 && @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    if args[:loss] == :linear
        args[:loss] = Linear()
    elseif args[:loss] == :logistic
        args[:loss] = Logistic()
    elseif args[:loss] == :gamma
        args[:loss] = Gamma()
    elseif args[:loss] == :tweedie
        args[:loss] = Tweedie()
    elseif args[:loss] == :L1
        args[:loss] = L1()
    elseif args[:loss] == :quantile
        args[:loss] = Quantile()
    else
        error("Invalid loss: $(args[:loss]). Only [`:linear`, `:logistic`, `:L1`, `:quantile`] are supported at the moment by EvoTreeRegressor.")
    end

    args[:rng] = mk_rng(args[:rng])::Random.AbstractRNG

    model = EvoTreeRegressor(
        args[:loss],
        args[:nrounds],
        args[:T](args[:lambda]),
        args[:T](args[:gamma]),
        args[:T](args[:eta]),
        args[:max_depth],
        args[:T](args[:min_weight]),
        args[:T](args[:rowsample]),
        args[:T](args[:colsample]),
        args[:nbins],
        args[:T](args[:alpha]),
        args[:monotone_constraints],
        args[:metric],
        args[:rng],
        args[:device])

    return model
end


mutable struct EvoTreeCount{T<:AbstractFloat,U<:ModelType,S<:Int} <: MMI.Probabilistic
    loss::U
    nrounds::S
    lambda::T
    gamma::T
    eta::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    alpha::T
    monotone_constraints
    metric::Symbol
    rng
    device
end

function EvoTreeCount(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float64,
        :loss => :poisson,
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
        :metric => :mse,
        :rng => 123,
        :device => "cpu"
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 && @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 && @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    if args[:loss] != :poisson
        error("Invalid loss: $(args[:loss]). Only `:poisson` is supported by EvoTreeCount.")
    else
        args[:loss] = Poisson()
    end

    args[:rng] = mk_rng(args[:rng])::Random.AbstractRNG

    model = EvoTreeCount(
        args[:loss],
        args[:nrounds],
        args[:T](args[:lambda]),
        args[:T](args[:gamma]),
        args[:T](args[:eta]),
        args[:max_depth],
        args[:T](args[:min_weight]),
        args[:T](args[:rowsample]),
        args[:T](args[:colsample]),
        args[:nbins],
        args[:T](args[:alpha]),
        args[:monotone_constraints],
        args[:metric],
        args[:rng],
        args[:device])

    return model
end

mutable struct EvoTreeClassifier{T<:AbstractFloat,U<:ModelType,S<:Int} <: MMI.Probabilistic
    loss::U
    nrounds::S
    lambda::T
    gamma::T
    eta::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    alpha::T
    metric::Symbol
    rng
    device
end

function EvoTreeClassifier(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :T => Float64,
        :loss => :softmax,
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
        :metric => :mse,
        :rng => 123,
        :device => "cpu"
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 && @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 && @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    if args[:loss] != :softmax
        error("Invalid loss: $(args[:loss]). Only `:softmax` is supported by EvoTreeClassifier.")
    else
        args[:loss] = Softmax()
    end

    args[:rng] = mk_rng(args[:rng])::Random.AbstractRNG

    model = EvoTreeClassifier(
        args[:loss],
        args[:nrounds],
        args[:T](args[:lambda]),
        args[:T](args[:gamma]),
        args[:T](args[:eta]),
        args[:max_depth],
        args[:T](args[:min_weight]),
        args[:T](args[:rowsample]),
        args[:T](args[:colsample]),
        args[:nbins],
        args[:T](args[:alpha]),
        args[:metric],
        args[:rng],
        args[:device])

    return model
end

mutable struct EvoTreeGaussian{T<:AbstractFloat,U<:ModelType,S<:Int} <: MMI.Probabilistic
    loss::U
    nrounds::S
    lambda::T
    gamma::T
    eta::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    alpha::T
    monotone_constraints
    metric::Symbol
    rng
    device
end

function EvoTreeGaussian(; kwargs...)

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
        :metric => :mse,
        :rng => 123,
        :device => "cpu"
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 && @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 && @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    if args[:loss] != :gaussian
        error("Invalid loss: $(args[:loss]). Only `:gaussian` is supported by EvoTreeGaussian.")
    else
        args[:loss] = Gaussian()
    end

    args[:rng] = mk_rng(args[:rng])::Random.AbstractRNG

    model = EvoTreeGaussian(
        args[:loss],
        args[:nrounds],
        args[:T](args[:lambda]),
        args[:T](args[:gamma]),
        args[:T](args[:eta]),
        args[:max_depth],
        args[:T](args[:min_weight]),
        args[:T](args[:rowsample]),
        args[:T](args[:colsample]),
        args[:nbins],
        args[:T](args[:alpha]),
        args[:monotone_constraints],
        args[:metric],
        args[:rng],
        args[:device])

    return model
end


# For R-package
function EvoTreeRModels(
    loss,
    nrounds,
    lambda,
    gamma,
    eta,
    max_depth,
    min_weight,
    rowsample,
    colsample,
    nbins,
    alpha,
    metric,
    rng,
    device)

    rng = mk_rng(rng)::Random.AbstractRNG

    if loss ∈ [:linear, :L1, :logistic, :quantile]
        if loss == :linear
            model_type = Linear()
        elseif loss == :logistic
            model_type = Logistic()
        elseif loss == :quantile
            model_type = Quantile()
        elseif loss == :L1
            model_type = L1()
        end
        model = EvoTreeRegressor(model_type, nrounds, Float32(lambda), Float32(gamma), Float32(eta), max_depth, Float32(min_weight), Float32(rowsample), Float32(colsample), nbins, Float32(alpha), metric, rng, device)
    elseif loss == :poisson
        model = EvoTreeCount(Poisson(), nrounds, Float32(lambda), Float32(gamma), Float32(eta), max_depth, Float32(min_weight), Float32(rowsample), Float32(colsample), nbins, Float32(alpha), metric, rng, device)
    elseif loss == :softmax
        model = EvoTreeClassifier(Softmax(), nrounds, Float32(lambda), Float32(gamma), Float32(eta), max_depth, Float32(min_weight), Float32(rowsample), Float32(colsample), nbins, Float32(alpha), metric, rng, device)
    elseif loss == :gaussian
        model = EvoTreeGaussian(Gaussian(), nrounds, Float64(lambda), Float64(gamma), Float64(eta), max_depth, Float64(min_weight), Float64(rowsample), Float64(colsample), nbins, Float64(alpha), metric, rng, device)
    else
        throw("invalid loss")
    end

    return model
end

# const EvoTypes = Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian}
const EvoTypes{T,U,S} = Union{EvoTreeRegressor{T,U,S},EvoTreeCount{T,U,S},EvoTreeClassifier{T,U,S},EvoTreeGaussian{T,U,S}}
