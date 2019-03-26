

# compute the gradient and hessian given target and predict
# linear
function update_grads!(::Val{:linear}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = 2 * (pred - target)
    @. δ² = pred * 0.0 + 2.0
end

# compute the gradient and hessian given target and predict
# logistic - on linear predictor
function update_grads!(::Val{:logistic}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = sigmoid(pred) * (1 - target) - (1 - sigmoid(pred)) * target
    @. δ² = sigmoid(pred) * (1 - sigmoid(pred))
end


# compute the gradient and hessian given target and predict
# logistic - directly on probs
# function update_grads!(::Val{:logistic}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}) where T <: AbstractFloat
#     @. δ = (1 - target) / (1 - pred) - target / pred
#     @. δ² = (1 - target) / (1 - pred) ^ 2 + target / pred ^ 2
# end


# compute the gradient and hessian given target and predict
# logistic
function logit(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = x / (1 - x)
    return δ, δ²
end

function sigmoid(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = exp(x) / (1 + exp(x))
    return x
end

function sigmoid(x::AbstractFloat)
    x = exp(x) / (1 + exp(x))
    return x
end

# # compute the gradient and hessian given target and predict
# function grad_hess(pred::AbstractArray{T}, target::AbstractArray{T}, loss::logistic) where {T<:AbstractFloat}
#     δ = 2 * (pred - target)
#     δ² = ones(size(pred)) * 2.0
#     return δ, δ²
# end

function update_gains!(info::SplitInfo{T}, ∑δL::T, ∑δ²L::T, ∑δR::T, ∑δ²R::T, λ::T) where T <: AbstractFloat
    info.gainL = (∑δL ^ 2 / (∑δ²L + λ)) / 2.0
    info.gainR = (∑δR ^ 2 / (∑δ²R + λ)) / 2.0
end

function update_track!(track::SplitTrack{T}, λ::T) where T <: AbstractFloat
    track.gainL = (track.∑δL ^ 2 / (track.∑δ²L + λ)) / 2.0
    track.gainR = (track.∑δR ^ 2 / (track.∑δ²R + λ)) / 2.0
    track.gain = track.gainL + track.gainR
end

# Calculate the gain for a given split
function get_gain(∑δ::T, ∑δ²::T, λ::T) where T <: AbstractFloat
    gain = (∑δ ^ 2 / (∑δ² + λ)) / 2.0
    return gain
end
