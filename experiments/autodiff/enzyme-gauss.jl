using EvoTrees
using EvoTrees: GaussianMLE, update_grads!
using Enzyme
using Statistics: mean, std, cor

nobs = 3
p = zeros(2, nobs)
p[1, :] = [0.0, 1.0, 3.0]
y = ones(nobs)
w = ones(nobs)

##################################
# reference
##################################
∇ = zeros(5, nobs)
∇[end, :] = w
config = EvoTree
update_grads!(∇, p, y, GaussianMLE, EvoTreeMLE())
∇

##################################
# enzyme
##################################
function gaussian_mle(p, y)
    loss = 0.0
    for i in eachindex(y)
        loss += (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return loss
end

# first orders reconcile with current implementation
dp = zeros(size(p))
autodiff(Reverse, gaussian_mle, Active, Duplicated(p, dp), Const(y))
dp

function grad(p, dp, y)
    Enzyme.autodiff(Reverse, gaussian_mle, Active, Duplicated(p, dp), Const(y))
    return nothing
end
dp = zeros(size(p))
vp = ones(size(p))
hess = zeros(size(p))
Enzyme.autodiff(Enzyme.Forward, grad,
    Enzyme.Duplicated(p, vp),
    Enzyme.Duplicated(dp, hess),
    Const(y),
)
dp
vp
hess


##################################
# enzyme - single point
##################################
p = [0.0, 0.0]
y = 1.0
function gaussian_mle(p, y)
    (p[2] + (y - p[1])^2 / (2 * exp(2 * p[2])))
end

# first orders reconcile with current implementation
dp = zeros(size(p))
autodiff(Reverse, gaussian_mle, Active, Duplicated(p, dp), Const(y))
dp

function grad(p, dp, y)
    Enzyme.autodiff(Reverse, gaussian_mle, Active, Duplicated(p, dp), Const(y))
    return nothing
end
dp = zeros(2, 2)
vp = ones(size(p))
hess = zeros(2, 2)
Enzyme.autodiff(Enzyme.Forward, grad,
    Enzyme.Duplicated(p, vp),
    Enzyme.Duplicated(dp, hess),
    Const(y),
)
dp
vp
hess


##################################
# enzyme - Hessian from JuMP
##################################
f(x::T...) where {T} = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

function enzyme_∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
    # direction(i) returns a tuple with a `1` in the `i`'th entry and `0`
    # otherwise
    direction(i) = ntuple(j -> Enzyme.Active(T(i == j)), N)
    # As the inner function, compute the gradient using Reverse mode
    ∇f(x...) = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, x...)[1]
    # For the outer autodiff, use Forward mode.
    hess = Enzyme.autodiff(
        Enzyme.Forward,
        ∇f,
        # Compute multiple evaluations of Forward mode, each time using `x` but
        # initializing with a different direction.
        Enzyme.BatchDuplicated.(Enzyme.Active.(x), ntuple(direction, N))...,
    )[1]
    # Unpack Enzyme's `hess` data structure into the matrix `H` expected by
    # JuMP.
    for j in 1:N, i in 1:j
        H[j, i] = hess[j][i]
    end
    return
end

function enzyme_∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
    # direction(i) returns a tuple with a `1` in the `i`'th entry and `0`
    # otherwise
    direction(i) = ntuple(j -> Enzyme.Active(T(i == j)), N)
    # As the inner function, compute the gradient using Reverse mode
    ∇f(x...) = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, x...)[1]
    # For the outer autodiff, use Forward mode.
    hess = Enzyme.autodiff(
        Enzyme.Forward,
        ∇f,
        # Compute multiple evaluations of Forward mode, each time using `x` but
        # initializing with a different direction.
        Enzyme.BatchDuplicated.(Enzyme.Active.(x), ntuple(direction, N))...,
    )[1]
    return hess
end

enzyme_∇²f(enzyme_H, x...)

x = rand(2)
enzyme_H = zeros(2, 2)
enzyme_∇²f(enzyme_H, x...)

direction(i) = ntuple(j -> Enzyme.Active(Float64(i == j)), 3)
direction(3)
