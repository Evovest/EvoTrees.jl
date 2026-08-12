@inline _cuda_target(y::CuDeviceVector, k, i) = y[i]
@inline _cuda_target(y::CuDeviceMatrix, k, i) = y[k, i]

#####################
# MAE
#####################
function kernel_mae_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            ∇[k, i] = (_cuda_target(y, k, i) - p[k, i]) * w
            ∇[K+k, i] = zero(T)
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::Union{CuVector,CuMatrix},
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_mae_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Credibility
#####################
function kernel_cred_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            d = _cuda_target(y, k, i) - p[k, i]
            ∇[k, i] = d * w
            ∇[K+k, i] = d^2 * w
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::Union{CuVector,CuMatrix},
    ::Type{<:EvoTrees.Cred},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_cred_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Quantile
#####################
function kernel_quantile_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y, alpha::T) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            diff = _cuda_target(y, k, i) - p[k, i]
            ∇[k, i] = diff > 0 ? alpha * w : (alpha - 1) * w
            ∇[K+k, i] = diff
        end
    end
    return
end
function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::Union{CuVector{T},CuMatrix{T}},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_quantile_∇!(∇, p, y, T(params.alpha))
    CUDA.synchronize()
    return
end

#####################
# MultiQuantile
#####################
function kernel_multiquantile_∇!(
    ∇::CuDeviceMatrix{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceVector{T},
    alphas::CuDeviceVector{T},
    K::Int,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        yi = y[i]
        w_idx = 2 * K + 1
        @inbounds wi = ∇[w_idx, i]
        @inbounds for k in 1:K
            diff = yi - p[k, i]
            alpha = alphas[k]
            ∇[k, i] = diff > 0 ? alpha * wi : (alpha - 1) * wi
            ∇[K + k, i] = diff
        end
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.MultiQuantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
) where {T<:AbstractFloat}
    K = length(params.alphas)
    alphas = CuArray(T.(params.alphas))
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_multiquantile_∇!(∇, p, y, alphas, K)
    CUDA.synchronize()
    return
end

#####################
# Softmax
#####################
function kernel_mlogloss_∇!(∇::CuDeviceMatrix{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector) where {T}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K = size(p, 1)
    if i <= length(y)
        isum = zero(T)
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
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_mlogloss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
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
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_gauss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_gauss_mt_∇!(∇, p, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
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
    return
end

function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.GaussianMLE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks=blocks threads=threads kernel_gauss_mt_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_logistic_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
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
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogisticMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_logistic_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

function kernel_logistic_mt_∇!(∇, p, y)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
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
    return
end

function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.LogisticMLE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks=blocks threads=threads kernel_logistic_mt_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

################################################################################
# StudentT (location-scale, ν fixed) - mirrors mle2p_grad_hess(::Type{StudentMLE}, ...)
# in src/loss.jl.
#
# h_μ is the FISHER information inv*(ν+1)/(ν+3), not the observed hessian - the observed one
# goes negative for |z| > √ν and get_gain divides by ∑h.
#
# ν -> Inf collapses every line onto kernel_gauss_∇!: a -> 1, (ν+1)/(ν+3) -> 1,
# ν(ν+1)/(ν+u)² -> 1. phase0-parity.jl test 2 is exactly that check.
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_student_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector, ν)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(y)
        w = ∇[5, i]
        μ = p[1, i]
        ls = p[2, i]
        inv = exp(-2 * ls)
        d = μ - y[i]
        u = d^2 * inv
        a = (ν + 1) / (ν + u)
        ∇[1, i] = d * inv * a * w
        ∇[2, i] = (1 - u * a) * w
        ∇[3, i] = inv * (ν + 1) / (ν + 3) * w
        ∇[4, i] = 2 * u * ν * (ν + 1) / (ν + u)^2 * w
    end
    return
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.StudentMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024,
)
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    # eltype(p), NOT Float64. A Float64 literal captured in a Float32 kernel silently promotes
    # every expression it touches and costs a large fraction of the throughput.
    ν = eltype(p)(EvoTrees._nu(params))
    @cuda blocks = blocks threads = threads kernel_student_∇!(∇, p, y, ν)
    CUDA.synchronize()
    return
end

function kernel_student_mt_∇!(∇, p, y, ν)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        w_row = 4 * Y + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for t in 1:Y
            gb = 2 * (t - 1)
            hb = 2 * Y + 2 * (t - 1)
            μ = p[2t-1, i]
            ls = p[2t, i]
            inv = exp(-2 * ls)
            d = μ - y[t, i]
            u = d^2 * inv
            a = (ν + 1) / (ν + u)
            ∇[gb+1, i] = d * inv * a * w
            ∇[gb+2, i] = (1 - u * a) * w
            ∇[hb+1, i] = inv * (ν + 1) / (ν + 3) * w
            ∇[hb+2, i] = 2 * u * ν * (ν + 1) / (ν + u)^2 * w
        end
    end
    return
end

function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::CuMatrix, ::Type{EvoTrees.StudentMLE}, params::EvoTrees.EvoTypes; MAX_THREADS=1024)
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    ν = eltype(p)(EvoTrees._nu(params))
    @cuda blocks=blocks threads=threads kernel_student_mt_∇!(∇, p, y, ν)
    CUDA.synchronize()
    return
end

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

function kernel_gradreg_∇!(∇, p, y, ::Type{L}) where {L<:EvoTrees.GradientRegression}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(p, 2)
        K = size(p, 1)
        w_row = 2 * K + 1
        @inbounds w = ∇[w_row, i]
        @inbounds for k in 1:K
            g, h = gradreg_grad_hess(L, p[k, i], _cuda_target(y, k, i))
            ∇[k, i] = g * w
            ∇[K+k, i] = h * w
        end
    end
    return
end
function EvoTrees.update_grads!(∇::CuMatrix, p::CuMatrix, y::Union{CuVector,CuMatrix}, ::Type{L}, params::EvoTrees.EvoTypes; MAX_THREADS=1024) where {L<:EvoTrees.GradientRegression}
    threads = min(MAX_THREADS, size(p, 2))
    blocks = cld(size(p, 2), threads)
    @cuda blocks = blocks threads = threads kernel_gradreg_∇!(∇, p, y, L)
    CUDA.synchronize()
    return
end
