#####################
# MAE
#####################
@kernel function kernel_mae_∇!(∇, @Const(p), @Const(y))
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            g, h = EvoTrees.mae_grad_hess(p[k, i], EvoTrees._target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
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
            g, h = EvoTrees.cred_grad_hess(p[k, i], EvoTrees._target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
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
            g, diff = EvoTrees.quantile_grad_diff(p[k, i], EvoTrees._target(y, k, i), alpha)
            ∇[k, i] = g * w
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
            g, diff = EvoTrees.quantile_grad_diff(p[k, i], yi, alphas[k])
            ∇[k, i] = g * wi
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
        w = ∇[end, i]
        for k in 1:K
            g, h = EvoTrees.mlogloss_grad_hess(p[k, i], isum, k == y[i])
            ∇[k, i] = g * w
            ∇[k+K, i] = h * w
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
# Two-parameter MLE (gaussian_mle, logistic_mle)
#
# One kernel covers single- and multi-target: single target is simply Y == 1,
# which gives gb == 0, hb == 2 and w == ∇[5, i] -- the layout the previous
# dedicated single-target kernels wrote by hand.
################################################################################
@kernel function kernel_mle2p_∇!(∇, @Const(p), @Const(y), ::Type{L}) where {L<:EvoTrees.MLE2P}
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        Y = size(p, 1) ÷ 2
        w = ∇[4*Y+1, i]
        for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            g1, g2, h1, h2 = EvoTrees.mle2p_grad_hess(
                L, p[2t-1, i], p[2t, i], EvoTrees._target(y, t, i)
            )
            ∇[gb+1, i] = g1 * w
            ∇[gb+2, i] = g2 * w
            ∇[hb+1, i] = h1 * w
            ∇[hb+2, i] = h2 * w
        end
    end
end
function EvoTrees.update_grads!(
    ∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix},
    ::Type{L}, params::EvoTrees.EvoTypes;
) where {L<:EvoTrees.MLE2P}
    backend = get_backend(∇)
    kernel_mle2p_∇!(backend)(∇, p, y, L; ndrange=size(p, 2))
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# GradientRegression (mse, logloss, poisson, gamma, tweedie)
#####################
@kernel function kernel_gradreg_∇!(∇, @Const(p), @Const(y), ::Type{L}) where {L<:EvoTrees.GradientRegression}
    i = @index(Global, Linear)
    @inbounds if i <= size(p, 2)
        K = size(p, 1)
        w = ∇[2*K+1, i]
        for k in 1:K
            g, h = EvoTrees.gradreg_grad_hess(L, p[k, i], EvoTrees._target(y, k, i))
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
