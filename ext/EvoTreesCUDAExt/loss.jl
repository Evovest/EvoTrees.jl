using KernelAbstractions

@kernel function kernel_mse_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MSE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_mse_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_mae_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_mae_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_cred_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = (y[i] - p[1, i]) * ∇[3, i]
        @inbounds ∇[2, i] = (y[i] - p[1, i])^2 * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{<:EvoTrees.Cred},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_cred_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_quantile_∇!(∇, p, y, alpha)
    i = @index(Global)
    if i <= length(y)
        diff = (y[i] - p[1, i])
        @inbounds ∇[1, i] = diff > 0 ? alpha * ∇[3, i] : (alpha - 1) * ∇[3, i]
        @inbounds ∇[2, i] = diff
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
) where {T<:AbstractFloat}
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_quantile_∇!(backend)(∇, p, y, T(params.alpha); ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_logloss_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = EvoTrees.sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_logloss_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_poisson_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Poisson},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_poisson_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_gamma_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Gamma},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_gamma_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_tweedie_∇!(∇, p, y)
    i = @index(Global)
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Tweedie},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_tweedie_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

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
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MLogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_mlogloss_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_gauss_∇!(∇, p, y)
    i = @index(Global)
    @inbounds if i <= length(y)
        μ = p[1, i]
        MIN_LOG_SIGMA = eltype(p)(-8.0)
        log_σ = max(p[2, i], MIN_LOG_SIGMA)
        σ_sq = exp(2 * log_σ)

        diff = μ - y[i]
        ∇[1, i] = diff / σ_sq * ∇[5, i]
        ∇[2, i] = (1 - (diff * diff) / σ_sq) * ∇[5, i]
        
        ∇[3, i] = ∇[5, i] / σ_sq
        ∇[4, i] = 2 * ∇[5, i] * (diff * diff) / σ_sq
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.GaussianMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_gauss_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function kernel_logistic_mle_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds begin
            
            ∇[1, i] = -tanh((y[i] - p[1, i]) / (2 * exp(p[2, i]))) * exp(-p[2, i]) * ∇[5, i]
            ∇[2, i] = -(
                exp(-p[2, i]) *
                (y[i] - p[1, i]) *
                tanh((y[i] - p[1, i]) / (2 * exp(p[2, i]))) - 1
            ) * ∇[5, i]
            
            half_z = (y[i] - p[1, i]) / (2 * exp(p[2, i]))
            inv_s2_half = 1 / (2 * exp(2 * p[2, i]))
            cosh_half = cosh(half_z)
            sech2_half = 1 / (cosh_half * cosh_half)
            ∇[3, i] = sech2_half * inv_s2_half * ∇[5, i]
            arg = exp(-p[2, i]) * (p[1, i] - y[i])
            num = exp(-2 * p[2, i]) * (p[1, i] - y[i]) * (p[1, i] - y[i] + exp(p[2, i]) * sinh(arg))
            den = 1 + cosh(arg)
            ∇[4, i] = (num / den) * ∇[5, i]
        end
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogisticMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = get_backend(p)
    threads = min(MAX_THREADS, length(y))
    kernel_logistic_mle_∇!(backend)(∇, p, y; ndrange=length(y), workgroupsize=threads)
    KernelAbstractions.synchronize(backend)
    return
end
