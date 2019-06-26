# define an abstrat tree node type - concrete types are TreeSplit and TreeLeaf
abstract type Node{T<:AbstractFloat} end

abstract type ModelType end
abstract type GradientRegression <: ModelType end
abstract type L1Regression <: ModelType end
abstract type QuantileRegression <: ModelType end
struct Linear <: GradientRegression end
struct Poisson <: GradientRegression end
struct Logistic <: GradientRegression end
struct L1 <: L1Regression end
struct Quantile <: QuantileRegression end

# compact alternative to ModeLData - not used for now
# To Do: how to exploit pre-sorting and binning
struct TrainData{T<:AbstractFloat}
    X::Matrix{T}
    X_permsort::Matrix{T}
    Y::Matrix{T}
    δ::Vector{T}
    δ²::Vector{T}
    𝑤::Vector{T}
end

mutable struct SplitInfo{T<:AbstractFloat, S<:Int}
    gain::T
    ∑δL::T
    ∑δ²L::T
    ∑𝑤L::T
    ∑δR::T
    ∑δ²R::T
    ∑𝑤R::T
    gainL::T
    gainR::T
    𝑖::S
    feat::S
    cond::T
end

mutable struct SplitTrack{T<:AbstractFloat}
    ∑δL::T
    ∑δ²L::T
    ∑𝑤L::T
    ∑δR::T
    ∑δ²R::T
    ∑𝑤R::T
    gainL::T
    gainR::T
    gain::T
end

struct TreeNode{T<:AbstractFloat, S<:Int, B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    pred::T
    split::B
end

TreeNode(left::S, right::S, feat::S, cond::T) where {T<:AbstractFloat, S<:Int} = TreeNode{T,S,Bool}(left, right, feat, cond, 0.0, true)
TreeNode(pred::T) where {T<:AbstractFloat} = TreeNode{T,Int,Bool}(0, 0, 0, 0.0, pred, false)

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
    elseif loss == :poisson model_type = Poisson()
    elseif loss == :L1 model_type = L1()
    elseif loss == :quantile model_type = Quantile()
    end

    model = EvoTreeRegressor(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)
    # message = MLJBase.clean!(model)
    # isempty(message) || @warn message
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
    end

    model = EvoTreeRegressor(model_type, nrounds, λ, γ, η, max_depth, min_weight, rowsample, colsample, nbins, α, metric, seed)
    # message = MLJBase.clean!(model)
    # isempty(message) || @warn message
    return model
end

# single tree is made of a root node that containes nested nodes and leafs
struct TrainNode{T<:AbstractFloat, I<:BitSet, J<:AbstractArray{Int, 1}, S<:Int}
    depth::S
    ∑δ::T
    ∑δ²::T
    ∑𝑤::T
    gain::T
    𝑖::I
    𝑗::J
end

# single tree is made of a root node that containes nested nodes and leafs
struct Tree{T<:AbstractFloat, S<:Int}
    nodes::Vector{TreeNode{T,S,Bool}}
end

# eval metric tracking
struct Metric
    iter::Vector{Int}
    metric::Vector{Float64}
end
Metric() = Metric([0], [Inf])

# gradient-boosted tree is formed by a vector of trees
struct GBTree{T<:AbstractFloat, S<:Int}
    trees::Vector{Tree{T,S}}
    params::EvoTreeRegressor
    metric::Metric
end
