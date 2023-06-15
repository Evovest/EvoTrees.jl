#####################
# linear
#####################
function kernel_linear_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
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
    ::EvoTreeRegressor{L,T};
    MAX_THREADS=1024
) where {L<:Linear,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_linear_∇!(∇, p, y)
    CUDA.synchronize()
    return
end

#####################
# Logistic
#####################
function kernel_logistic_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
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
    ::EvoTreeRegressor{L,T};
    MAX_THREADS=1024
) where {L<:Logistic,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_logistic_∇!(∇, p, y)
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
    ::EvoTreeCount{L,T};
    MAX_THREADS=1024
) where {L<:Poisson,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
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
    ::EvoTreeRegressor{L,T};
    MAX_THREADS=1024
) where {L<:Gamma,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
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
    ::EvoTreeRegressor{L,T};
    MAX_THREADS=1024
) where {L<:Tweedie,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_tweedie_∇!(∇, p, y)
    CUDA.synchronize()
    return
end


#####################
# Softmax
#####################
function kernel_softmax_∇!(∇::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K = (size(∇, 1) - 1) ÷ 2
    if i <= length(y)
        @inbounds for k = 1:K
            if k == y[i]
                ∇[k, i] = (p[k, i] - 1) * ∇[2*K+1, i]
            else
                ∇[k, i] = p[k, i] * ∇[2*K+1, i]
            end
            ∇[k+K, i] = (1 - p[k, i]) * ∇[2*K+1, i]
        end
    end
    return
end
function update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::EvoTreeClassifier{L,T};
    MAX_THREADS=1024
) where {L<:MultiClassRegression,T}
    p_prob = exp.(p) ./ sum(exp.(p), dims=1)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_softmax_∇!(∇, p_prob, y)
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
    ::Union{EvoTreeGaussian{L,T},EvoTreeMLE{L,T}};
    MAX_THREADS=1024
) where {L<:GaussianMLE,T}
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_gauss_∇!(∇, p, y)
    CUDA.synchronize()
    return
end