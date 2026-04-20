# Usage of Offset


Offset terms allow incorporating prior knowledge or the output of an existing model directly into the boosted tree training process. Rather than learning the full response from scratch, EvoTrees learns only the **residual**: the incremental signal beyond what the offset already explains.

This tutorial demonstrates offset usage on the Titanic survival dataset. A linear logistic regression (GLM) using only `Age` and `Sex` is fitted first as a baseline. Its linear predictions (logits) are then passed as an offset to EvoTrees, which learns to capture the remaining variation from the other features. Comparing feature importances between the two EvoTrees models reveals how the offset absorbs the linear signal, leaving the trees to focus on non-linear structure and interactions.

## Key Principles

- **EvoTrees learns a residual over the offset**: When an offset is provided, offset is used as an initial predictions from which the model will learn. Gradients are therefore computed relative to this shifted baseline, and the trees only learn what the offset does not already explain.

- **Predictions do not include the offset**: Calling `model(data)` returns only the tree-based component (the residual). To recover full predictions, you must manually integrate the offsets and the residual prediction. This may require to to first apply link (e.g., logits for `logloss`) to combine the offset and residual prediction, prior the projecting back to the natural basis using the inverse link function.

- **Stacking requires care**: Because offset is excluded from `model(data)` outputs, any downstream model in a stacking pipeline must receive the complete combined signal (offset plus the residual model prediction), rather than just the raw EvoTree output alone.

## Getting Started

Load the required packages and the dataset:

``` julia
using EvoTrees
using EvoTrees: logit, sigmoid
using GLM
using MLDatasets
using DataFrames
using Statistics: mean, median
using CategoricalArrays
using Random

df = MLDatasets.Titanic().dataframe
```

## Preprocessing

Prepare features into a model-compatible format. EvoTrees’ Tables API supports `Real` (including `Bool`) and `Categorical` inputs. String variables such as `Sex` are converted to unordered `Categorical`. Missing values in `Age` are handled by first creating a missingness indicator, then imputing with the median.

``` julia
# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age)

# remove columns not used as features
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]
```

Split into train and evaluation sets, and define target and feature names. Note that `feature_names` is captured here, before any offset column is added to the dataframe.

``` julia
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

``` julia
config = EvoTreeRegressor(
    loss=:logloss,
    nrounds=200,
    early_stopping_rounds=10,
    eta=0.05,
    max_depth=5,
    rowsample=0.5,
    colsample=0.9)

model = EvoTrees.fit(
    config, dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10)
```

``` julia
mean((model(dtrain) .> 0.5) .== dtrain[!, target_name])
```

    0.8779803646563815

``` julia
mean((model(deval) .> 0.5) .== deval[!, target_name])
```

    0.8258426966292135

Feature importance shows `Sex` and `Age` as the dominant predictors, consistent with the well-known survival patterns on the Titanic:

``` julia
EvoTrees.importance(model)
```

    7-element Vector{Pair{Symbol, Float64}}:
               :Sex => 0.3618707154011808
               :Age => 0.22239138290345634
              :Fare => 0.2152044075062655
            :Pclass => 0.12022692955736314
             :SibSp => 0.05037551228606886
             :Parch => 0.019528664105300648
     :Age_ismissing => 0.01040238824036474

## Baseline GLM Model

We now fit a vanilla logistic regression using only `Age` and `Sex`. This linear model captures the strong, well-understood survival gradient along those two dimensions.

``` julia
glm_model = glm(@formula(Survived ~ Sex + Age), dtrain, Binomial(), LogitLink())
```

We extract the predictions for both (`dtrain` and `deval`). GLM returns probabilities by default, which is the natural basis expected by the offset input.

``` julia
# logit(p) = log(p / (1 - p))
offset_train = predict(glm_model, dtrain)
offset_eval = predict(glm_model, deval)
```

Add the offset columns to the DataFrames. Since `feature_names` was defined before this step, it does not include `"offset"` and EvoTrees will not use it as a predictor feature.

``` julia
dtrain[!, :offset] = offset_train
deval[!, :offset] = offset_eval
```

``` julia
mean((offset_train .> 0.5) .== dtrain[!, target_name])
```

    0.7896213183730715

``` julia
mean((offset_eval .> 0.5) .== deval[!, target_name])
```

    0.7752808988764045

## EvoTrees With GLM Offset

We train a new EvoTrees model with the same configuration and the same full set of features, but now passing the GLM logits as an offset via `offset_name`. The model’s gradients are computed relative to the GLM baseline, so the trees only learn the residual signal not already captured by the linear model.

``` julia
model_offset = EvoTrees.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    offset_name="offset",
    print_every_n=10)
```

## Diagnosis

### Feature Importance Comparison

``` julia
EvoTrees.importance(model_offset)
```

    7-element Vector{Pair{Symbol, Float64}}:
              :Fare => 0.30798676125890256
               :Age => 0.28327502947633404
            :Pclass => 0.19407926280644047
               :Sex => 0.09063675270901962
             :SibSp => 0.06618870002666856
             :Parch => 0.04098912223088763
     :Age_ismissing => 0.016844371491747123

Feature importance from the offset model shows that the importance of `Sex` has significantly shrinked, while `Age` has maintained a strong importance. The residual EvoTree model captures the non-linear and interaction effect missed by the GLM, as well as effect from the additional features. For `Sex`, since it’s a binary feature, much of its signal was already captured by the GLM.

### Full Predictions

Raw predictions from the offset model **do not include the offset**. They represent only the tree-based residual component:

``` julia
# residual component only (offset NOT included)
pred_residual_train = model_offset(dtrain)
pred_residual_eval = model_offset(deval)
```

To recover full probabilities, combine the offset logits with the residual logits on the **link scale**, then apply the inverse link (sigmoid):

``` julia
# Full predictions: offset logits + residual logits, then sigmoid
full_pred_train = sigmoid.(logit.(offset_train) .+ logit.(pred_residual_train))
full_pred_eval = sigmoid.(logit.(offset_eval) .+ logit.(pred_residual_eval))
```

``` julia
mean((full_pred_train .> 0.5) .== dtrain[!, target_name])
```

    0.8821879382889201

``` julia
mean((full_pred_eval .> 0.5) .== deval[!, target_name])
```

    0.8202247191011236

Accuracy is comparable to the no-offset model, confirming that the GLM offset absorbs the dominant linear signal while the trees efficiently handle the remaining nonlinear structure.
