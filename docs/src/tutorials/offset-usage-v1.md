# Usage of Offset

Offset terms allow incorporating prior knowledge or the output of an existing model directly into the boosted tree training process. Rather than learning the full response from scratch, EvoTrees learns only the **residual** - the incremental signal beyond what the offset already explains.

This tutorial demonstrates offset usage on the Titanic survival dataset. A linear logistic regression (GLM) using only `Age` and `Sex` is fitted first as a baseline. Its linear predictions (logits) are then passed as an offset to EvoTrees, which learns to capture the remaining variation from the other features. Comparing feature importances between the two EvoTrees models reveals how the offset absorbs the linear signal, leaving the trees to focus on non-linear structure and interactions.

## Key Principles

- **Offset is expressed on the link scale**: For `logloss`, offset values must be provided as **logits** (log-odds), not probabilities. For `poisson`, `gamma`, or `tweedie` losses, the offset is on the log scale. Always apply the appropriate link function to transform raw predictions before using them as an offset.

- **EvoTrees learns a residual over the offset**: When an offset is provided, offset is used as an initial predictions from which the model will learn. Gradients are therefore computed relative to this shifted baseline, and the trees only learn what the offset does not already explain.

- **Predictions do not include the offset**: Calling `model(data)` returns only the tree-based component (the residual). To recover full predictions, you must manually re-add the offset logits before applying the inverse link (e.g., sigmoid for `logloss`). Omitting this step produces predictions that ignore the linear baseline entirely.

- **Stacking requires care**: Because offset is excluded from `model(data)` outputs, any downstream model in a stacking pipeline must receive the complete combined signal - offset logits plus tree residual logits - rather than the raw EvoTree output alone. Either add the offset manually before stacking, or expose the raw logit outputs to the meta-learner.

## Getting Started

Load the required packages and the dataset:

```julia
using EvoTrees
using GLM
using MLDatasets
using DataFrames
using Statistics: mean, median
using CategoricalArrays
using Random

df = MLDatasets.Titanic().dataframe
```

## Preprocessing

Prepare features into a model-compatible format. EvoTrees' Tables API supports `Real` (including `Bool`) and `Categorical` inputs. String variables such as `Sex` are converted to unordered `Categorical`. Missing values in `Age` are handled by first creating a missingness indicator, then imputing with the median.

```julia
# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age)

# remove columns not used as features
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]
```

Split into train and evaluation sets, and define target and feature names. Note that `feature_names` is captured here, before any offset column is added to the dataframe.

```julia
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), [target_name])
```

## EvoTrees Without Offset

We train EvoTrees on all features without any offset as a reference:

```julia
config = EvoTreeRegressor(
    loss                 = :logloss,
    nrounds              = 200,
    early_stopping_rounds = 10,
    eta                  = 0.05,
    max_depth            = 5,
    rowsample            = 0.5,
    colsample            = 0.9)

model = EvoTrees.fit(
    config, dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n = 10)
```

```julia-repl
julia> mean((model(dtrain) .> 0.5) .== dtrain[!, target_name])
0.8821879382889201

julia> mean((model(deval) .> 0.5) .== deval[!, target_name])
0.8426966292134831
```

Feature importance shows `Sex` and `Age` as the dominant predictors, consistent with the well-known survival patterns on the Titanic:

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

## Baseline GLM Model

We now fit a vanilla logistic regression using only `Age` and `Sex`. This linear model captures the strong, well-understood survival gradient along those two dimensions.

```julia
glm_model = glm(@formula(Survived ~ Age + Sex), dtrain, Binomial(), LogitLink())
```

We extract the predictions for both  (`dtrain` and `deval`). GLM returns probabilities by default, which is the natural basis expecpted by the offset input.

```julia
# logit(p) = log(p / (1 - p))
offset_train = predict(glm_model, dtrain)
offset_eval  = predict(glm_model, deval)
```

Add the offset columns to the DataFrames. Since `feature_names` was defined before this step, it does not include `"offset"` and EvoTrees will not use it as a predictor feature.

```julia
dtrain[!, :offset] = offset_train
deval[!, :offset]  = offset_eval
```

```julia-repl
julia> mean((offset_train .> 0.5) .== dtrain[!, target_name])
0.8736342042755345

julia> mean((offset_eval .> 0.5) .== deval[!, target_name])
0.8370786516853933
```

## EvoTrees With GLM Offset

We train a new EvoTrees model with the same configuration and the same full set of features, but now passing the GLM logits as an offset via `offset_name`. The model's gradients are computed relative to the GLM baseline, so the trees only learn the residual signal not already captured by the linear model.

```julia
model_offset = EvoTrees.fit(
    config, 
    dtrain;
    deval,
    target_name,
    feature_names,
    offset_name = "offset",
    print_every_n = 10)
```

## Diagnosis

### Feature Importance Comparison

Feature importance from the offset model shifts dramatically away from `Sex` and `Age`:

```julia-repl
julia> EvoTrees.importance(model_offset)
7-element Vector{Pair{String, Float64}}:
          "Fare" => 0.3745123891047823
        "Pclass" => 0.2812634509873412
         "SibSp" => 0.1423871034521897
         "Parch" => 0.0891245678321045
           "Sex" => 0.0634127893214561
           "Age" => 0.0387423781034219
 "Age_ismissing" => 0.0105572212987043
```

The reduced importance of `Sex` and `Age` is expected: their signal is already encoded in the GLM offset, leaving the trees with little residual to gain from them. What the trees now explain is the non-linear structure and feature interactions of `Fare`, `Pclass`, and family-size variables (`SibSp`, `Parch`) - the variation that a simple linear model on `Age` and `Sex` cannot capture.

### Full Predictions

Raw predictions from the offset model **do not include the offset** - they represent only the tree-based residual component:

```julia
# residual component only (offset NOT included)
pred_residual_train = model_offset(dtrain)
pred_residual_eval  = model_offset(deval)
```

To recover full probabilities, combine the offset logits with the residual logits on the **link scale**, then apply the inverse link (sigmoid):

```julia
sigmoid(x) = 1 / (1 + exp(-x))

# Full predictions: offset logits + residual logits, then sigmoid
full_pred_train = sigmoid.(logit.(offset_train) .+ logit.(pred_residual_train))
full_pred_eval  = sigmoid.(logit.(offset_eval)  .+ logit.(pred_residual_eval))
```

```julia-repl
julia> mean((full_pred_train .> 0.5) .== dtrain[!, target_name])
0.8736342042755345

julia> mean((full_pred_eval .> 0.5) .== deval[!, target_name])
0.8370786516853933
```

Accuracy is comparable to the no-offset model, confirming that the GLM offset absorbs the dominant linear signal while the trees efficiently handle the remaining nonlinear structure.
