#####################
# MSE
#####################
function kernel_mse_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds ∇[1, i] = 2 * (p[i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeRegressor{L};
    MAX_THREADS=1024
) where {L<:MSE}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_mse_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Logistic
#####################
function kernel_logloss_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeRegressor{L};
    MAX_THREADS=1024
) where {L<:LogLoss}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_logloss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Poisson
#####################
function kernel_poisson_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeCount{L};
    MAX_THREADS=1024
) where {L<:Poisson}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_poisson_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Gamma
#####################
function kernel_gamma_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeRegressor{L};
    MAX_THREADS=1024
) where {L<:Gamma}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_gamma_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Tweedie
#####################
function kernel_tweedie_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] =
            2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeRegressor{L};
    MAX_THREADS=1024
) where {L<:Tweedie}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_tweedie_∇!(∇, p, y)
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

function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeClassifier{L};
    MAX_THREADS=1024
) where {L<:MLogLoss}
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

function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Union{EvoTreeGaussian{L},EvoTreeMLE{L}};
    MAX_THREADS=1024
) where {L<:GaussianMLE}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads kernel_gauss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end