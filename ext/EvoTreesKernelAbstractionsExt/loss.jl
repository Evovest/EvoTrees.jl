# device-array target accessors (backend-agnostic)
@inline _dev_target(y::AbstractVector, k, i) = @inbounds y[i]
@inline _dev_target(y::AbstractMatrix, k, i) = @inbounds y[k, i]

#####################
# MAE
#####################
@kernel function kernel_mae_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            ∇[k, i] = (_dev_target(y, k, i) - p[k, i]) * w
            ∇[K+k, i] = zero(eltype(∇))
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix},
    ::Type{EvoTrees.MAE}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_mae_∇!(backend)(∇, p, y; ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Credibility
#####################
@kernel function kernel_cred_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            d = _dev_target(y, k, i) - p[k, i]
            ∇[k, i] = d * w
            ∇[K+k, i] = d^2 * w
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix},
    ::Type{<:EvoTrees.Cred}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_cred_∇!(backend)(∇, p, y; ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Quantile
#####################
@kernel function kernel_quantile_∇!(∇, @Const(p), @Const(y), alpha)
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            diff = _dev_target(y, k, i) - p[k, i]
            ∇[k, i] = diff > 0 ? alpha * w : (alpha - 1) * w
            ∇[K+k, i] = diff
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T}, p::CuMatrix{T}, y::Union{CuVector{T},CuMatrix{T}},
    ::Type{EvoTrees.Quantile}, params::EvoTrees.EvoTypes;
) where {T<:AbstractFloat}
    backend = get_backend(∇)
    kernel_quantile_∇!(backend)(∇, p, y, T(params.alpha); ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# MultiQuantile
#####################
@kernel function kernel_multiquantile_∇!(∇, @Const(p), @Const(y), @Const(alphas), K::Int)
    i = @index(Global, Linear)
    @inbounds if i <= length(y)
        yi = y[i]
        wi = ∇[2*K+1, i]
        for k in 1:K
            diff = yi - p[k, i]
            alpha = alphas[k]
            ∇[k, i] = diff > 0 ? alpha * wi : (alpha - 1) * wi
            ∇[K+k, i] = diff
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T}, p::CuMatrix{T}, y::CuVector{T},
    ::Type{EvoTrees.MultiQuantile}, params::EvoTrees.EvoTypes;
) where {T<:AbstractFloat}
    K = length(params.alphas)
    backend = get_backend(∇)
    alphas = _to_device(backend, T.(params.alphas))
    kernel_multiquantile_∇!(backend)(∇, p, y, alphas, K; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Softmax (MLogLoss)
#####################
@kernel function kernel_mlogloss_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    T = eltype(∇)
    K = size(p, 1)
    @inbounds if i <= length(y)
        isum = zero(T)
        for k in 1:K
            isum += exp(p[k, i])
        end
        for k in 1:K
            iexp = exp(p[k, i])
            if k == y[i]
                ∇[k, i] = (iexp / isum - 1) * ∇[end, i]
            else
                ∇[k, i] = iexp / isum * ∇[end, i]
            end
            ∇[k+K, i] = 1 / isum * (1 - iexp / isum) * ∇[end, i]
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::CuVector,
    ::Type{EvoTrees.MLogLoss}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_mlogloss_∇!(backend)(∇, p, y; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return
end

################################################################################
# Gaussian MLE - single target: pred[1]=μ, pred[2]=log(σ)
################################################################################
@kernel function kernel_gauss_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= length(y)
        ∇[1, i] = (p[1, i] - y[i]) / exp(2 * p[2, i]) * ∇[5, i]
        ∇[2, i] = (1 - (p[1, i] - y[i])^2 / exp(2 * p[2, i])) * ∇[5, i]
        ∇[3, i] = ∇[5, i] / exp(2 * p[2, i])
        ∇[4, i] = 2 * ∇[5, i] / exp(2 * p[2, i]) * (p[1, i] - y[i])^2
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::CuVector,
    ::Type{EvoTrees.GaussianMLE}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_gauss_∇!(backend)(∇, p, y; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return
end

# Gaussian MLE - multi-target
@kernel function kernel_gauss_mt_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w = ∇[4*Y+1, i]
        for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            inv = 1 / exp(2 * ls)
            d = μ - yt
            ∇[gb+1, i] = d * inv * w
            ∇[gb+2, i] = (1 - d^2 * inv) * w
            ∇[hb+1, i] = inv * w
            ∇[hb+2, i] = 2 * inv * d^2 * w
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::CuMatrix,
    ::Type{EvoTrees.GaussianMLE}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_gauss_mt_∇!(backend)(∇, p, y; ndrange=size(y, 2))
    KernelAbstractions.synchronize(backend)
    return
end

# Logistic MLE - single target
@kernel function kernel_logistic_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= length(y)
        w = ∇[5, i]
        μ = p[1, i]
        ls = p[2, i]
        yt = y[i]
        ∇[1, i] = -tanh((yt - μ) / (2 * exp(ls))) * exp(-ls) * w
        ∇[2, i] = -(exp(-ls) * (yt - μ) * tanh((yt - μ) / (2 * exp(ls))) - 1) * w
        ∇[3, i] = sech((yt - μ) / (2 * exp(ls)))^2 / (2 * exp(2 * ls)) * w
        ∇[4, i] = (exp(-2 * ls) * (μ - yt) *
                   (μ - yt + exp(ls) * sinh(exp(-ls) * (μ - yt)))) /
                  (1 + cosh(exp(-ls) * (μ - yt))) * w
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::CuVector,
    ::Type{EvoTrees.LogisticMLE}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_logistic_∇!(backend)(∇, p, y; ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    return
end

# Logistic MLE - multi-target
@kernel function kernel_logistic_mt_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w = ∇[4*Y+1, i]
        for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            ∇[gb+1, i] = -tanh((yt - μ) / (2 * exp(ls))) * exp(-ls) * w
            ∇[gb+2, i] = -(exp(-ls) * (yt - μ) * tanh((yt - μ) / (2 * exp(ls))) - 1) * w
            ∇[hb+1, i] = sech((yt - μ) / (2 * exp(ls)))^2 / (2 * exp(2 * ls)) * w
            ∇[hb+2, i] = (exp(-2 * ls) * (μ - yt) *
                          (μ - yt + exp(ls) * sinh(exp(-ls) * (μ - yt)))) /
                         (1 + cosh(exp(-ls) * (μ - yt))) * w
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::CuMatrix,
    ::Type{EvoTrees.LogisticMLE}, params::EvoTrees.EvoTypes;
)
    backend = get_backend(∇)
    kernel_logistic_mt_∇!(backend)(∇, p, y; ndrange=size(y, 2))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# GradientRegression (mse, logloss, poisson, gamma, tweedie)
#####################
@inline gradreg_grad_hess(::Type{EvoTrees.MSE}, pk, yk) = (2 * (pk - yk), 2 * one(pk))
@inline function gradreg_grad_hess(::Type{EvoTrees.LogLoss}, pk, yk)
    pred = EvoTrees.sigmoid(pk)
    return (pred - yk, pred * (1 - pred))
end
@inline gradreg_grad_hess(::Type{EvoTrees.Poisson}, pk, yk) = (exp(pk) - yk, exp(pk))
@inline function gradreg_grad_hess(::Type{EvoTrees.Gamma}, pk, yk)
    pred = exp(pk)
    return (2 * (1 - yk / pred), 2 * yk / pred)
end
@inline function gradreg_grad_hess(::Type{EvoTrees.Tweedie}, pk, yk)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return (
        2 * (pred^(2 - rho) - yk * pred^(1 - rho)),
        2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * yk * pred^(1 - rho)),
    )
end

@kernel function kernel_gradreg_∇!(∇, @Const(p), @Const(y), ::Type{L}) where {L<:EvoTrees.GradientRegression}
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            g, h = gradreg_grad_hess(L, p[k, i], _dev_target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix},
    ::Type{L}, params::EvoTrees.EvoTypes;
) where {L<:EvoTrees.GradientRegression}
    backend = get_backend(∇)
    kernel_gradreg_∇!(backend)(∇, p, y, L; ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return
end