# Logistic Regression on Titanic Dataset

We will use the Titanic dataset, which is included in the MLDatasets package. It describes the survival status of individual passengers on the Titanic. The model will be approached as a logistic regression problem, although a Classifier model could also have been used (see the `Classification - Iris` tutorial). 

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

df = MLDatasets.Titanic().dataframe
```

## Preprocessing

Before we can train our model, we need to preprocess the dataset. We will split our data according to train and eval indices, and separate features from the target variable.

```julia
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

# treat string feature and missing values
transform!(df, :Sex => ByRow(x -> x == "male" ? 0 : 1) => :Sex)
transform!(df, :Age => ByRow(x -> ismissing(x) ? mean(skipmissing(df.Age)) : x) => :Age)

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, Not(:Survived)]), train_data[:, :Survived]
x_eval, y_eval = Matrix(eval_data[:, Not(:Survived)]), eval_data[:, :Survived]
```

## Training

Now we are ready to train our model. We will first define a model configuration using the [`EvoTreeRegressor`](@ref) model constructor. 
Then, we'll use [`fit_evotree`](@ref) to train a boosted tree model. We'll pass optional `x_eval` and `y_eval` arguments, which enable the usage of early stopping. 

```julia
config = EvoTreeRegressor(loss = :logistic, nrounds=200, eta=0.1, max_depth=5, rowsample = 0.6, colsample = 0.9)
model = fit_evotree(config;
    x_train, y_train,
    x_eval, y_eval,
    metric = :logloss,
    early_stopping_rounds=10,
    print_every_n=10)
```

Finally, we can get predictions by passing training and testing data to our model. We can then evaluate the accuracy of our model, which should be around 85%. 

```julia
pred_train = model(x_train)
pred_eval = model(x_eval)
```

```julia-repl
julia> mean((pred_train .> 0.5) .== y_train)
0.8835904628330996

julia> mean((pred_eval .> 0.5) .== y_eval)
0.8370786516853933
```
