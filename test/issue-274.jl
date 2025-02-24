using CUDA
using EvoTrees
using CategoricalArrays

x_train = x=rand(10, 2)
y_train = categorical(rand("abc", 10), ordered=true)
config = EvoTreeClassifier(; device=:gpu, L2=123)
m = fit(config; x_train, y_train)
@time yhat = m(x_train; device=:gpu)
