# Ranking with Yahoo! Learning to Rank Challenge. 

In this tutorial, we present how a ranking task can be tackled using regular regression techniques without compromising performance compared to specialized ranking learners.
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
Then, we use [`EvoTrees.fit`](@ref) to train a boosted tree model. The optional `x_eval` and `y_eval` arguments are provided to enable the usage of early stopping. 

```julia
config = EvoTreeRegressor(
    nrounds=6000,
    early_stopping_rounds=200,
    loss=:mse,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

m_mse, logger_mse = EvoTrees.fit(
    config;
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval,
    print_every_n=50,
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

## Using native group support

The section above computes NDCG after the fact, outside the model. EvoTrees can also be told
about groups directly, which makes two things available during training.

Pass the query id alongside the features and target. `group_train` and `group_eval` take one
id per row, and rows sharing an id form a group. Ids need not be contiguous or sorted, so the
query vectors above can be passed as they are:

```julia
config = EvoTreeRegressor(
    nrounds=6000,
    early_stopping_rounds=200,
    loss=:mse,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
    metric=:ndcg,
    ndcg_k=10,
)

m = EvoTrees.fit(
    config;
    x_train, y_train, group_train=q_train,
    x_eval, y_eval, group_eval=q_eval,
    print_every_n=50,
);
```

This changes two behaviours.

`metric = :ndcg` computes NDCG within each group and averages over groups, which is the same
quantity the `groupby` above produces, so early stopping now selects on ranking quality
rather than on squared error. `ndcg_k` sets the truncation rank, and defaults to scoring the
full list.

`rowsample` becomes group aware: whole groups are sampled rather than individual rows, so a
query is never split across the sampled and unsampled sets. This matters because a group is
the unit NDCG is defined over, and a partial group changes the comparison set.

When fitting from a table, the equivalent is `group_name`, and the column is excluded from
the inferred features. `dtrain` above is the LIBSVM parse result rather than a table, so build
one from the arrays already extracted:

```julia
df_train = DataFrame(x_train, :auto)
df_train.y, df_train.q = y_train, q_train
df_eval = DataFrame(x_eval, :auto)
df_eval.y, df_eval.q = y_eval, q_eval

m = EvoTrees.fit(config, df_train; target_name="y", group_name="q", deval=df_eval)
```

Groups are supported on both CPU and GPU.

## Weights and grouped metrics

A group's weight is the mean of its rows' weights, so the default of unit weights leaves every
query equally weighted regardless of how many documents it holds. Giving a query's rows a
common weight of 2 makes that query count twice one whose rows weigh 1.

Only that group-level weight reaches `:ndcg`. NDCG is defined from the ranking of a group's
documents, so the spread of weights *within* a group is deliberately ignored, which matches the
canonical definition and the per-query weights other ranking libraries accept.

Where per-document weights should count, `metric = :corr` scores the weighted Pearson
correlation between prediction and target within each group and averages over groups. A row's
own weight enters its group's correlation, and the group weighs by the mean of its rows'
weights. Groups of fewer than two rows, and groups whose target is constant, carry no signal
and are left out of the average; a group whose prediction is constant while its target is not
scores zero.

A grouped metric does not require grouped training. `eval_group_name` sets the group column for
`deval` alone, so a model can train with ordinary per-row sampling while a group-aware metric is
tracked:

```julia
config = EvoTreeRegressor(loss=:mse, metric=:corr)
m = EvoTrees.fit(config, df_train; target_name="y", eval_group_name="q", deval=df_eval)
```

It defaults to `group_name`, so grouping both sides stays a single argument.

## Ranking objective

With groups available, `loss=:lambdarank` optimises ranking directly rather than fitting the
relevance as a regression target. Pairs of documents within a query contribute a pairwise
cost weighted by the NDCG change that swapping them would cause, so the model is trained on
within-query order rather than on absolute relevance. `metric` defaults to `:ndcg`.

```julia
config = EvoTreeRegressor(
    loss=:lambdarank,
    ndcg_k=10,
    nrounds=6000,
    early_stopping_rounds=200,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

m = EvoTrees.fit(
    config;
    x_train, y_train, group_train=q_train,
    x_eval, y_eval, group_eval=q_eval,
    print_every_n=50,
);
```

Relevance must be non-negative, since gains are `2^rel - 1`. Cost is quadratic in the size
of a query, which is not a concern at typical query sizes but is worth knowing if a single
group is very large.

Groups are passed to `EvoTrees.fit` directly. The MLJ interface has no way to carry them,
so ranking is not available through it.

Whether this beats the regression approach above depends on the data. It helps most where
queries differ in how they are graded, since the absolute label level is then query-specific
noise that a regression fit must absorb and a ranking objective can ignore. Where relevance
is comparable across queries, regression is already a strong baseline, which is what the
opening of this tutorial reports.

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
    early_stopping_rounds=200,
    loss=:logloss,
    eta=0.01,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

m_logloss, logger_logloss = EvoTrees.fit(
    config;
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval,
    print_every_n=50,
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

We've seen that a ranking problem can be efficiently handled with generic regression tasks, yet achieve comparable performance to specialized ranking loss functions. Below, we present the NDCG obtained from the above experiments along those published on CatBoost's [benchmarks](https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results).


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
