using EvoTrees
using Statistics: mean, std, cor
using DifferentiationInterface
import Enzyme
import Mooncake

###################################
# x^2
###################################
function mse(x)
    loss_mse = sum(x .^ 2)
    return loss_mse
end

x = [0.0, 1.0, 2.0, 3.0, 4.0]
backend = AutoEnzyme()
gradient(mse, backend, x)

# backend = SecondOrder(AutoEnzyme(; mode=Enzyme.EnzymeCore.Forward), AutoEnzyme(; mode=Enzyme.EnzymeCore.Reverse))
backend = SecondOrder(AutoEnzyme(), AutoEnzyme())
# backend = SecondOrder(AutoForwardDiff(), AutoZygote())
# backend = SecondOrder(AutoMooncake(; config=Mooncake.Config()), AutoMooncake(; config=Mooncake.Config()))
hvp(mse, backend, x)
hessian(mse, backend, x)
value_gradient_and_hessian(mse, backend, x)

###################################
# mse
###################################
function mse(x, y)
    loss_mse = sum((x .- y) .^ 2)
    return loss_mse
end

x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = ones(5)
dx = zeros(5)

backend = SecondOrder(AutoEnzyme(), AutoEnzyme())
# backend = SecondOrder(AutoForwardDiff(), AutoZygote())
gradient(f_sparse_scalar, backend, x)

###################################
# Test with penalty
###################################
function mse_ctrl(x, y, z)
    loss_mse = sum((x .- y) .^ 2)
    loss_corr = cor(x, z) .* length(x)
    return loss_mse + loss_corr
end

x = [0.0, 1.0, 2.0, 3.0, 4.0]
# x = [6.5, 6.0, 5.5, 5.0, 4.0]

# y = ones(5)
y = copy(x)

z = [0.0, 1.0, 1.0, 1.0, 2.0]
# z = copy(x)
dx = zeros(5)
autodiff(Reverse, mse_ctrl, Active, Duplicated(x, dx), Const(y), Const(z))
dx

function grad_ctrl(x, dx, y, z)
    Enzyme.autodiff(Reverse, mse_ctrl, Active, Duplicated(x, dx), Const(y), Const(z))
    return nothing
end
dx = zeros(5)
vx = ones(5)
hess = zeros(5)
Enzyme.autodiff(Enzyme.Forward, grad_ctrl,
    Enzyme.Duplicated(x, vx),
    Enzyme.Duplicated(dx, hess),
    Const(y),
    Const(z)
)
dx
vx
hess

###################################
# Test with third degree
###################################
mse3(x, y) = sum((x .- y) .^ 3)
function grad3(x, dx, y)
    Enzyme.autodiff(Reverse, mse3, Active, Duplicated(x, dx), Const(y))
    return nothing
end
x = [4.0]
y = [1.0]
dx = zeros(1)
vx = ([1.0], [1.0])
hess = ([0.0], [0.0])
Enzyme.autodiff(Enzyme.Forward, grad3,
    Enzyme.BatchDuplicated(x, vx),
    Enzyme.BatchDuplicated(dx, hess),
    Const(y)
)
dx
vx
hess
