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

A first step in data processing is to prepare the input features in a model compatible format. 

EvoTrees' Tables API supports input that are either `Real`, `Bool` or `Categorical`.
A recommended approach for `String` features such as `Sex` is to convert them into an unordered `Categorical`. 

For dealing with features withh missing values such as `Age`, a common approach is to first create an `Bool` indicator variable capturing the info on whether a value is missing.
Then, the missing values can be inputed (replaced by some default values such as `mean` or `median`, or more sophisticated approach such as predictions from another model).

```julia
# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

```

The full data can now be split according to train and eval indices. 
Target and feature names are also set.

```julia
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
fnames = setdiff(names(df), [target_name])
```

## Training

Now we are ready to train our model. We will first define a model configuration using the [`EvoTreeRegressor`](@ref) model constructor. 
Then, we'll use [`fit_evotree`](@ref) to train a boosted tree model. We'll pass optional `deval` arguments, which enables the tracking of an evaluation metric and early stopping. 

```julia
config = EvoTreeRegressor(
  loss=:logistic, 
  nrounds=200, 
  eta=0.05, 
  nbins=128, 
  max_depth=5, 
  rowsample=0.5, 
  colsample=0.9)

model = fit_evotree(
    config, dtrain; 
    deval,
    target_name,
    fnames,
    metric = :logloss,
    early_stopping_rounds=10,
    print_every_n=10)
```


## Diagnosis

We can get predictions by passing training and testing data to our model. We can then evaluate the accuracy of our model, which should be around 85%. 

```julia
pred_train = model(dtrain)
pred_eval = model(deval)
```

```julia-repl
julia> mean((pred_train .> 0.5) .== dtrain[!, target_name])
0.8821879382889201

julia> mean((pred_eval .> 0.5) .== deval[!, target_name])
0.8426966292134831
```

Finally, features importance can be inspected using [`EvoTrees.importance`](@ref).

```julia-repl
julia> EvoTrees.importance(model)
7-element Vector{Pair{String, Float64}}:
           "Sex" => 0.29612654189959403
           "Age" => 0.25487324307720827
          "Fare" => 0.2530947969323613
        "Pclass" => 0.11354283043193575
         "SibSp" => 0.05129209383816148
         "Parch" => 0.017385183317069588
 "Age_ismissing" => 0.013685310503669728
```
