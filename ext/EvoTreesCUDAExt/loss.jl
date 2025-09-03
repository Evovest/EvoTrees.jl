#####################
# MSE
#####################
@kernel function kernel_mse_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MSE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mse_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# LogLoss
#####################
@kernel function kernel_logloss_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = EvoTrees.sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_logloss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Poisson
#####################
@kernel function kernel_poisson_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Poisson},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_poisson_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Gamma
#####################
@kernel function kernel_gamma_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Gamma},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_gamma_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Tweedie
#####################
@kernel function kernel_tweedie_∇!(∇, p, y)
    i = @index(Global)
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Tweedie},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_tweedie_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Softmax
#####################
@kernel function kernel_mlogloss_∇!(∇, p, y)
    i = @index(Global)
    K = size(p, 1)
    if i <= length(y)
        isum = zero(eltype(p))
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds for k in 1:K
            iexp = exp(p[k, i])
            if k == y[i]
                ∇[k, i] = (iexp / isum - 1) * ∇[end, i]
            else
                ∇[k, i] = iexp / isum * ∇[end, i]
            end
            ∇[k+K, i] = 1 / isum * (1 - iexp / isum) * ∇[end, i]
        end
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MLogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mlogloss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
@kernel function kernel_gauss_∇!(∇, p, y)
    i = @index(Global)
    @inbounds if i <= length(y)
        # first order gradients
        ∇[1, i] = (p[1, i] - y[i]) / exp(2 * p[2, i]) * ∇[5, i]
        ∇[2, i] = (1 - (p[1, i] - y[i])^2 / exp(2 * p[2, i])) * ∇[5, i]
        # # second order gradients
        ∇[3, i] = ∇[5, i] / exp(2 * p[2, i])
        ∇[4, i] = 2 * ∇[5, i] / exp(2 * p[2, i]) * (p[1, i] - y[i])^2
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.GaussianMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_gauss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# MAE
#####################
@kernel function kernel_mae_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mae_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Credibility
#####################
@kernel function kernel_cred_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        @inbounds ∇[2, i] = (y[i] - p[1, i])^2 * ∇[3, i]
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{<:EvoTrees.Cred},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_cred_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Quantile
#####################
@kernel function kernel_quantile_∇!(∇, p, y, alpha)
    i = @index(Global)
    if i <= length(y)
        diff = (y[i] - p[1, i])
        @inbounds ∇[1, i] = diff > 0 ? alpha * ∇[3, i] : (alpha - 1) * ∇[3, i]
        @inbounds ∇[2, i] = diff
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_quantile_∇!(backend)(∇, p, y, T(params.alpha); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

