module EvoTreesDifferentiationInterfaceExt

using EvoTrees
using DifferentiationInterface: AbstractADType, value_derivative_and_second_derivative

@inline function EvoTrees.custom_grad_hess(loss_fn::F, backend::AbstractADType, pk, yk) where {F}
    _, g, h = value_derivative_and_second_derivative(Base.Fix2(loss_fn, yk), backend, pk)
    return (g, h)
end

end
