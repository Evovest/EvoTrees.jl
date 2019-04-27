# Explicit
pred = [-2, -1, 0, 1, 2]
y = [0, 0, 0, 0, 0]

loss = (y - pred) .^ 2
grad = 2*pred



# AutoDiff
using Flux
using Flux.Tracker

pred = param([-2, -1, 0, 1, 2])
y = [0, 0, 0, 0, 0]
loss = (y - pred) .^ 2

# backprop
Tracker.back!(sum(loss))

pred.data
pred.grad
