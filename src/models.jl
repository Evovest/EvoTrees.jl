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

mutable struct EvoTreeRegressor{T<:AbstractFloat, U<:ModelType, S<:Int} <: MLJBase.Deterministic
    loss::U
    nrounds::S
    λ::T
    γ::T
    η::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    α::T
    metric::Symbol
    seed::S
end

function EvoTreeRegressor(;
    loss=:linear,
    nrounds=10,
    λ=0.0, #
    γ=0.0, # gamma: min gain to split
    η=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    α=0.5,
    metric=:mse,
    seed=444)

    if loss == :linear model_type = Linear()
    elseif loss == :logistic model_type = Logistic()
    elseif loss == :L1 model_type = L1()
    elseif loss == :quantile model_type = Quantile()
    end

    model = EvoTreeRegressor(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)

    return model
end



mutable struct EvoTreeCount{T<:AbstractFloat, U<:ModelType, S<:Int} <: MLJBase.Deterministic
    loss::U
    nrounds::S
    λ::T
    γ::T
    η::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    α::T
    metric::Symbol
    seed::S
end

function EvoTreeCount(;
    loss=:linear,
    nrounds=10,
    λ=0.0, #
    γ=0.0, # gamma: min gain to split
    η=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    α=0.5,
    metric=:poisson,
    seed=444)

    model_type = Poisson()
    model = EvoTreeCount(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)

    return model
end


mutable struct EvoTreeClassifier{T<:AbstractFloat, U<:ModelType, S<:Int} <: MLJBase.Probabilistic
    loss::U
    nrounds::S
    λ::T
    γ::T
    η::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    α::T
    metric::Symbol
    seed::S
end

function EvoTreeClassifier(;
    loss=:linear,
    nrounds=10,
    λ=0.0, #
    γ=0.0, # gamma: min gain to split
    η=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    α=0.5,
    metric=:mlogloss,
    seed=444)

    model_type = Softmax()
    model = EvoTreeClassifier(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)

    return model
end



mutable struct EvoTreeGaussian{T<:AbstractFloat, U<:ModelType, S<:Int} <: MLJBase.Probabilistic
    loss::U
    nrounds::S
    λ::T
    γ::T
    η::T
    max_depth::S
    min_weight::T # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::T # subsample
    colsample::T
    nbins::S
    α::T
    metric::Symbol
    seed::S
end

function EvoTreeGaussian(;
    loss=:linear,
    nrounds=10,
    λ=0.0, #
    γ=0.0, # gamma: min gain to split
    η=0.1, # eta: learning rate
    max_depth=5,
    min_weight=1.0, # minimal weight, different from xgboost (but same for linear)
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    α=0.5,
    metric=:gaussian,
    seed=444)

    model_type = Gaussian()
    model = EvoTreeGaussian(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)

    return model
end


# For R-package
function EvoTreeRegressorR(
    loss,
    nrounds,
    λ,
    γ,
    η,
    max_depth,
    min_weight,
    rowsample,
    colsample,
    nbins,
    α,
    metric,
    seed)

    if loss == :linear model_type = Linear()
    elseif loss == :logistic model_type = Logistic()
    elseif loss == :poisson model_type = Poisson()
    elseif loss == :L1 model_type = L1()
    elseif loss == :quantile model_type = Quantile()
    elseif loss == :softmax model_type = Softmax()
    elseif loss == :gaussian model_type = Gaussian()
    end

    model = EvoTreeRegressor(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)

    return model
end

const EvoTypes = Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier}
