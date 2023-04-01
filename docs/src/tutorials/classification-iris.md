# Classication on Iris dataset

We will use the iris dataset, which is included in the MLDatasets package. This dataset consists of measurements of the sepal length, sepal width, petal length, and petal width for three different types of iris flowers: Setosa, Versicolor, and Virginica.

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

df = MLDatasets.Iris().dataframe
```

## Preprocessing

Before we can train our model, we need to preprocess the dataset. We will convert the class variable, which specifies the type of iris flower, into a categorical variable.

```julia
Random.seed!(123)

df[!, :class] = categorical(df[!, :class])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, 1:4]), train_data[:, :class]
x_eval, y_eval = Matrix(eval_data[:, 1:4]), eval_data[:, :class]
```

## Training

Now we are ready to train our model. We will first define a model configuration using the [`EvoTreeClassifier`](@ref) model constructor. 
Then, we'll use [`fit_evotree`](@ref) to train a boosted tree model. We'll pass optional `x_eval` and `y_eval` arguments, which enable the usage of early stopping. 

```julia
config = EvoTreeClassifier(nrounds=200, eta=0.1, max_depth=5, lambda=0.01, rowsample = 0.8)
model = fit_evotree(config;
    x_train, y_train,
    x_eval, y_eval,
    metric = :mlogloss,
    early_stopping_rounds=10,
    print_every_n=10)
```

Finally, we can get predictions by passing training and testing data to our model. We can then evaluate the accuracy of our model, which should be near 100% for this simple classification problem. 

```julia
pred_train = model(x_train)
idx_train = [findmax(row)[2] for row in eachrow(pred_train)]

pred_eval = model(x_eval)
idx_eval = [findmax(row)[2] for row in eachrow(pred_eval)]
```

```julia-repl
julia> mean(idx_eval .== levelcode.(y_eval))
1.0

julia> mean(idx_eval .== levelcode.(y_eval))
1.0
```
