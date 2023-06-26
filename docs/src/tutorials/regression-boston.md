# Regression on Boston Housing Dataset

We will use the Boston Housing dataset, which is included in the MLDatasets package. It's derived from information collected by the U.S. Census Service concerning housing in the area of Boston. Target variable represents the median housing value.

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

df = MLDatasets.BostonHousing().dataframe
```

## Preprocessing

Before we can train our model, we need to preprocess the dataset. We will split our data according to train and eval indices, and separate features from the target variable.

```julia
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, Not(:MEDV)]), train_data[:, :MEDV]
x_eval, y_eval = Matrix(eval_data[:, Not(:MEDV)]), eval_data[:, :MEDV]
```

## Training

Now we are ready to train our model. We will first define a model configuration using the [`EvoTreeRegressor`](@ref) model constructor. 
Then, we'll use [`fit_evotree`](@ref) to train a boosted tree model. We'll pass optional `x_eval` and `y_eval` arguments, which enable the usage of early stopping. 

```julia
config = EvoTreeRegressor(nrounds=200, eta=0.1, max_depth=4, lambda=0.1, rowsample = 0.9, colsample = 0.9)
model = fit_evotree(config;
    x_train, y_train,
    x_eval, y_eval,
    metric = :mse,
    early_stopping_rounds=10,
    print_every_n=10)
```

Finally, we can get predictions by passing training and testing data to our model. We can then apply various evaluation metric, such as the MAE (mean absolute error):  

```julia
pred_train = model(x_train)
pred_eval = model(x_eval)
```

```julia-repl
julia> mean(abs.(pred_train .- y_train))
1.056997874224627

julia> mean(abs.(pred_eval .- y_eval))
2.3298767665825264
```