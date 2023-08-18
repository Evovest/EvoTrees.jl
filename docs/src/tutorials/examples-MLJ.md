# MLJ Integration

EvoTrees.jl provides a first-class integration with the MLJ ecosystem. 

See [official project page](https://github.com/alan-turing-institute/MLJ.jl) for more info.

To use with MLJ, an EvoTrees model configuration must first be initialized using either: 
- [`EvoTreeRegressor`](@ref)
- [`EvoTreeClassifier`](@ref)
- [`EvoTreeCount`](@ref)
- [`EvoTreeMLE`](@ref)

The model is then passed to MLJ's `machine`, opening access to the rest of the MLJ modeling ecosystem. 

```julia
using StatsBase: sample
using EvoTrees
using EvoTrees: sigmoid, logit # only needed to create the synthetic data below
using MLJBase

features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = MLJBase.table(X)

# linear regression
tree_model = EvoTreeRegressor(loss=:linear, max_depth=5, eta=0.05, nrounds=10)

# set machine
mach = machine(tree_model, X, y)

# partition data
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split

# fit data
fit!(mach, rows=train, verbosity=1)

# continue training
mach.model.nrounds += 10
fit!(mach, rows=train, verbosity=1)

# predict on train data
pred_train = predict(mach, selectrows(X, train))
mean(abs.(pred_train - selectrows(Y, train)))

# predict on test data
pred_test = predict(mach, selectrows(X, test))
mean(abs.(pred_test - selectrows(Y, test)))
```