using EvoTrees
using EvoTrees: MSE, update_grads!
using Enzyme
using Statistics: mean, std, cor

# basic
nobs = 5
p = zeros(1, nobs)
p[1, :] = [0.0, 1.0, 2.0, 3.0, 4.0]
y = ones(nobs)
w = ones(nobs)

# rand
nobs = 1_000_000
p = rand(1, nobs)
y = rand(nobs)
w = ones(nobs)

##################################
# reference
##################################
∇ = zeros(3, nobs)
∇[end, :] = w
config = EvoTree
# 100k: 0.0003s
# 1M: 0.0015s
@time update_grads!(∇, p, y, MSE, EvoTreeRegressor())
∇

##################################
# enzyme
##################################
function enzyme_mse(p, y)
    loss = 0.0
    for i in eachindex(y)
        loss += (p[1, i] - y[i])^2
    end
    return loss
end

# first orders reconcile with current implementation
dp = zeros(size(p))
# 100k: 0.0006s
# 1M: 0.006s
@time autodiff(Reverse, enzyme_mse, Active, Duplicated(p, dp), Const(y))
dp

function _mse_grad(p, dp, y)
    Enzyme.autodiff(Reverse, enzyme_mse, Active, Duplicated(p, dp), Const(y))
    return nothing
end
dp = zeros(size(p))
vp = ones(size(p))
hess = zeros(size(p))

# 100k: 0.001s
# 1M: 0.015s
@time Enzyme.autodiff(Enzyme.Forward, _mse_grad,
    Enzyme.Duplicated(p, vp),
    Enzyme.Duplicated(dp, hess),
    Const(y),
)
dp
vp
hess
