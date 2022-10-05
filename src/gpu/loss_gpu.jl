# Gradient regression
function get_gain_gpu(::Type{L}, ∑::AbstractVector{T}, lambda::T) where {L<:GradientRegression,T<:AbstractFloat}
    gain = ∑[1]^2 / (∑[2] + lambda * ∑[3]) / 2
    return gain
end

# Gaussian regression
function get_gain_gpu(::Type{L}, ∑::AbstractVector{T}, lambda::T) where {L<:GaussianRegression,T<:AbstractFloat}
    gain = ∑[1]^2 / (∑[3] + lambda * ∑[5]) / 2 + ∑[2]^2 / (∑[4] + lambda * ∑[5]) / 2
    return gain
end

#####################
# linear
#####################
function kernel_linear_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds δ𝑤[1, i] = 2 * (p[i] - y[i]) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = 2 * δ𝑤[3, i]
    end
    return
end
function update_grads_gpu!(::Type{Linear}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_linear_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end

#####################
# Logistic
#####################
function kernel_logistic_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = sigmoid(p[1, i])
        @inbounds δ𝑤[1, i] = (pred - y[i]) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = pred * (1 - pred) * δ𝑤[3, i]
    end
    return
end
function update_grads_gpu!(::Type{Logistic}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_logistic_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end

#####################
# Poisson
#####################
function kernel_poisson_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds δ𝑤[1, i] = (pred - y[i]) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = pred * δ𝑤[3, i]
    end
    return
end
function update_grads_gpu!(::Type{Poisson}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_poisson_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end

#####################
# Gamma
#####################
function kernel_gamma_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds δ𝑤[1, i] = 2 * (1 - y[i] / pred) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = 2 * y[i] / pred * δ𝑤[3, i]
    end
    return
end
function update_grads_gpu!(::Type{Gamma}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_gamma_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end

#####################
# Tweedie
#####################
function kernel_tweedie_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds δ𝑤[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * δ𝑤[3, i]
        @inbounds δ𝑤[2, i] = 2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * δ𝑤[3, i]
    end
    return
end
function update_grads_gpu!(::Type{Tweedie}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_tweedie_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end


################################################################################
# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
################################################################################
function kernel_gauss_δ𝑤!(δ𝑤::CuDeviceMatrix, p::CuDeviceMatrix, y::CuDeviceVector)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(y)
        # first order gradients
        δ𝑤[1, i] = (p[1, i] - y[i]) / exp(2 * p[2, i]) * δ𝑤[5, i]
        δ𝑤[2, i] = (1 - (p[1, i] - y[i])^2 / exp(2 * p[2, i])) * δ𝑤[5, i]
        # second order gradients
        δ𝑤[3, i] = δ𝑤[5, i] / exp(2 * p[2, i])
        δ𝑤[4, i] = 2 * δ𝑤[5, i] / exp(2 * p[2, i]) * (p[1, i] - y[i])^2
    end
    return
end

function update_grads_gpu!(::Type{Gaussian}, δ𝑤::CuMatrix, p::CuMatrix, y::CuVector; MAX_THREADS=1024)
    threads = min(MAX_THREADS, length(y))
    blocks = ceil(Int, (length(y)) / threads)
    @cuda blocks = blocks threads = threads kernel_gauss_δ𝑤!(δ𝑤, p, y)
    CUDA.synchronize()
    return
end


function update_childs_∑_gpu!(::Type{L}, nodes, n, bin, feat) where {L}
    nodes[n<<1].∑ .= nodes[n].hL[:, bin, feat]
    nodes[n<<1+1].∑ .= nodes[n].hR[:, bin, feat]
    return nothing
end