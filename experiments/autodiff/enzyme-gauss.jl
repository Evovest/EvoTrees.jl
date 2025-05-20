using EvoTrees
using EvoTrees: GaussianMLE, update_grads!
using Enzyme
using Statistics: mean, std, cor


p = zeros(2, 5)
p[1, :] = [0.0, 1.0, 2.0, 3.0, 4.0]
y = ones(5)
w = ones(5)

##################################
# reference
##################################
∇ = zeros(5, 5)
∇[end, :] = w
config = EvoTree
update_grads!(∇, p, y, GaussianMLE, EvoTreeRegressor())
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
