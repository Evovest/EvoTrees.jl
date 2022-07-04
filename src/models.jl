abstract type ModelType end
abstract type GradientRegression <: ModelType end
abstract type L1Regression <: ModelType end
abstract type QuantileRegression <: ModelType end
abstract type MultiClassRegression <: ModelType end
abstract type GaussianRegression <: ModelType end
struct Linear <: GradientRegression end
struct Poisson <: GradientRegression end
struct Logistic <: GradientRegression end
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
    metric::Symbol
    rng
    device
end

function EvoTreeRegressor(;
    T::Type=Float64,
    loss=:linear,
    nrounds=10,
    lambda=0.0, #
    gamma=0.0, # gamma: min gain to split
    eta=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    alpha=0.5,
    metric=:mse,
    rng=123,
    device="cpu")

    if loss == :linear
        model_type = Linear()
    elseif loss == :logistic
        model_type = Logistic()
    elseif loss == :L1
        model_type = L1()
    elseif loss == :quantile
        model_type = Quantile()
    else
        error("Invalid loss: $loss. Only [`:linear`, `:logistic`, `:L1`, `:quantile`],  losses are supported at the moment by EvoTreeRegressor.")
    end

    rng = mk_rng(rng)::Random.AbstractRNG

    model = EvoTreeRegressor(model_type, nrounds, T(lambda), T(gamma), T(eta), max_depth, T(min_weight), T(rowsample), T(colsample), nbins, T(alpha), metric, rng, device)

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
    metric::Symbol
    rng
    device
end

function EvoTreeCount(;
    T::Type=Float64,
    loss=:poisson,
    nrounds=10,
    lambda=0.0, #
    gamma=0.0, # gamma: min gain to split
    eta=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    alpha=0.5,
    metric=:poisson,
    rng=123,
    device="cpu")

    rng = mk_rng(rng)::Random.AbstractRNG

    if loss != :poisson
        error("Invalid loss: $loss. Only `:poisson` loss is supported at the moment by EvoTreeCount.")
    end

    model = EvoTreeCount(Poisson(), nrounds, T(lambda), T(gamma), T(eta), max_depth, T(min_weight), T(rowsample), T(colsample), nbins, T(alpha), metric, rng, device)

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

function EvoTreeClassifier(;
    T::Type=Float64,
    loss=:softmax,
    nrounds=10,
    lambda=0.0, #
    gamma=0.0, # gamma: min gain to split
    eta=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    alpha=0.5,
    metric=:mlogloss,
    rng=123,
    device="cpu")

    rng = mk_rng(rng)::Random.AbstractRNG

    if loss != :softmax
        error("Invalid loss: $loss. Only `:softmax` loss is supported at the moment by EvoTreeClassifier.")
    end
    model = EvoTreeClassifier(Softmax(), nrounds, T(lambda), T(gamma), T(eta), max_depth, T(min_weight), T(rowsample), T(colsample), nbins, T(alpha), metric, rng, device)

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
    metric::Symbol
    rng
    device
end

function EvoTreeGaussian(;
    T::Type=Float64,
    loss=:gaussian,
    nrounds=10,
    lambda=0.0, #
    gamma=0.0, # gamma: min gain to split
    eta=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    alpha=0.5,
    metric=:gaussian,
    rng=123,
    device="cpu")

    rng = mk_rng(rng)::Random.AbstractRNG

    if loss != :gaussian
        error("Invalid loss: $loss. Only `:gaussian` loss is supported at the moment by EvoTreeGaussian.")
    end
    model = EvoTreeGaussian(Gaussian(), nrounds, T(lambda), T(gamma), T(eta), max_depth, T(min_weight), T(rowsample), T(colsample), nbins, T(alpha), metric, rng, device)

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
