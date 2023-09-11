# Ranking with Yahoo! Learning to Rank Challenge. 

In this tutorial, we we walk through how a ranking task can be tackled using regular regression techniques without compromise on performance compared to specialized ranking learners. 
The data used is from the `C14 - Yahoo! Learning to Rank Challenge`, which can be obtained following a request to [https://webscope.sandbox.yahoo.com](https://webscope.sandbox.yahoo.com).

## Getting started

To begin, we load the required packages:

```julia
using EvoTrees
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
```

## Load LIBSVM format data

Some datasets come in the so called `LIBSVM` format, which stores data using a sparse representation: 

```
<label> <query> <feature_id_1>:<feature_value_1> <feature_id_2>:<feature_value_2>
```

We use the [`ReadLIBSVM.jl`](https://github.com/jeremiedb/ReadLIBSVM.jl) package to perform parsing: 

```julia
using ReadLIBSVM
dtrain = read_libsvm("set1.train.txt"; has_query=true)
deval = read_libsvm("set1.valid.txt"; has_query=true)
dtest = read_libsvm("set1.test.txt"; has_query=true)
```

## Preprocessing

Preprocessing is minimal since all features are parsed as floats and specific files are provided for each of the train, eval and test splits. 

Several features are fully missing (contain only 0s) in the training dataset. They are removed from all datasets since they cannot bring value to the model.

Then, the features, targets and query ids are extracted from the parsed `LIBSVM` format. 

```julia
colsums_train = map(sum, eachcol(dtrain[:x]))
drop_cols = colsums_train .== 0

x_train = dtrain[:x][:, .!drop_cols]
x_eval = deval[:x][:, .!drop_cols]
x_test = dtest[:x][:, .!drop_cols]

# assign queries
q_train = dtrain[:q]
q_eval = deval[:q]
q_test = dtest[:q]

# assign targets
y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]
```

## Training

Now we are ready to train our model. We first define a model configuration using the [`EvoTreeRegressor`](@ref) model constructor. 
Then, we use [`fit_evotree`](@ref) to train a boosted tree model. The optional `x_eval` and `y_eval` arguments are provided to enable the usage of early stopping. 

```julia
config = EvoTreeRegressor(
    nrounds=6000,
    loss=:mse,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

m_mse, logger_mse = fit_evotree(
    config;
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval,
    early_stopping_rounds=200,
    print_every_n=50,
    metric=:mse,
    return_logger=true
);

p_test = m_mse(x_test);
```

## Model evaluation

For ranking problems, a commonly used metric is the [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain). It essentially considers whether the model is good at identifying the top K outcomes within a group. There are various flavors to its implementation, though the most commonly used one is the following:

```julia
function ndcg(p, y, k=10)
    k = min(k, length(p))
    p_order = partialsortperm(p, 1:k, rev=true)
    y_order = partialsortperm(y, 1:k, rev=true)
    _y = y[p_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    ndcg = sum(gains ./ discounts)

    y_order = partialsortperm(y, 1:k, rev=true)
    _y = y[y_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    idcg = sum(gains ./ discounts)
    return idcg == 0 ? 1.0 : ndcg / idcg
end
```

To compute the NDCG over a collection of groups, it is handy to leverage DataFrames' `combine` and `groupby` functionalities: 

```julia
test_df = DataFrame(p=p_test, y=y_test, q=q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = round(mean(test_df_agg.ndcg), sigdigits=5)
@info "ndcg_test MSE" ndcg_test

┌ Info: ndcg_test MSE
└   ndcg_test = 0.8008
```

## Logistic regression alternative

The above regression experiment shows a performance competitive with the results outlined in CatBoost's [ranking benchmarks](https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results). 

Another approach is to use a scaling of the the target ranking scores to perform a logistic regression.

```julia
max_rank = 4
y_train = dtrain[:y] ./ max_rank
y_eval = deval[:y] ./ max_rank
y_test = dtest[:y] ./ max_rank

config = EvoTreeRegressor(
    nrounds=6000,
    loss=:logloss,
    eta=0.01,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

m_logloss, logger_logloss = fit_evotree(
    config;
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval,
    early_stopping_rounds=200,
    print_every_n=50,
    metric=:logloss,
    return_logger=true
);
```

To measure the NDCG, the original targets must be used since NDCG is a scale sensitive measure.

```julia
y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

p_test = m_logloss(x_test);
test_df = DataFrame(p=p_test, y=y_test, q=q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = round(mean(test_df_agg.ndcg), sigdigits=5)
@info "ndcg_test LogLoss" ndcg_test

┌ Info: ndcg_test LogLoss
└   ndcg_test = 0.80267
```

## Conclusion

We've seen that a ranking problem can be efficiently handled with generic regression tasks, yet achieve comparable performance to specialized ranking loss functions. Below, we present the NDCG obtained from the above experiments along those presented by CatBoost's [benchmarks](https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results).


| **Model**               | **NDCG**  |
|-------------------------|-----------| 
| **EvoTrees - mse**      |**0.80080**|
| **EvoTrees - logistic** |**0.80267**|
| cat-rmse                |0.802115   | 
| cat-query-rmse          |0.802229   | 
| cat-pair-logit          |0.797318   | 
| cat-pair-logit-pairwise |0.790396   | 
| cat-yeti-rank           |0.802972   | 
| xgb-rmse                |0.798892   | 
| xgb-pairwise            |0.800048   | 
| xgb-lambdamart-ndcg     |0.800048   | 
| lgb-rmse                |0.8013675  | 
| lgb-pairwise            |0.801347   |


It should be noted that the later results were not reproduced in the scope of current tutorial, so one should be careful about any claim of model superiority. The results from CatBoost's benchmarks were however already indicative of strong performance of non-specialized ranking loss functions, to which this tutorial brings further support. 
