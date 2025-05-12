using EvoTrees
using Enzyme
using Statistics: mean, std, cor

# mse(x, y) = sum((x .- y) .^ 2)
function mse(x, y)
    loss_mse = sum((x .- y) .^ 2)
    return loss_mse
end

x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = ones(5)
dx = zeros(5)
autodiff(Reverse, mse, Active, Duplicated(x, dx), Const(y))
dx

function grad(x, dx, y)
    Enzyme.autodiff(Reverse, mse, Active, Duplicated(x, dx), Const(y))
    return nothing
end
dx = zeros(5)
vx = ones(5)
hess = zeros(5)
Enzyme.autodiff(Enzyme.Forward, grad,
    Enzyme.Duplicated(x, vx),
    Enzyme.Duplicated(dx, hess),
    Const(y),
)
dx
vx
hess

###################################
# Test with third degree
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
