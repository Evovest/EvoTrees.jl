# Ranking with Yahoo! Learning to Rank Challenge. 

In this ttutorial, we we walk through how a ranking task can be tackled using regular regression techniques without compromise on performance compared to specialised ranking learners. 
The data used is `C14 - Yahoo! Learning to Rank Challenge`, which can be obtained following a request to [https://webscope.sandbox.yahoo.com](https://webscope.sandbox.yahoo.com).

## Getting started

To begin, we will load the required packages:

```julia
using EvoTrees
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
```

## Load LibSVM format data

Some datastes come in so called `libsvm` format, which stores data using a sparse representation: 

```
<label> <query> <feature_id_1>:<feature_value_1> <feature_id_2>:<feature_value_2>
```

There was no known Julia package supporting the parsing of such storage format as the time of this tutorial, so the following simple parser was built:  

```julia
function read_libsvm(raw::Vector{UInt8}; has_query=false)

    io = IOBuffer(raw)
    lines = readlines(io)

    nobs = length(lines)
    nfeats = 0 # initialize number of features

    y = zeros(Float64, nobs)

    if has_query
        offset = 2 # offset for feature idx: y + query entries
        q = zeros(Int, nobs)
    else
        offset = 1 # offset for feature idx: y
    end

    vals = [Float64[] for _ in 1:nobs]
    feats = [Int[] for _ in 1:nobs]

    for i in eachindex(lines)
        line = lines[i]
        line_split = split(line, " ")

        y[i] = parse(Int, line_split[1])
        has_query ? q[i] = parse(Int, split(line_split[2], ":")[2]) : nothing

        n = length(line_split) - offset
        lfeats = zeros(Int, n)
        lvals = zeros(Float64, n)
        @inbounds for jdx in 1:n
            ls = split(line_split[jdx+offset], ":")
            lvals[jdx] = parse(Float64, ls[2])
            lfeats[jdx] = parse(Int, ls[1])
            lfeats[jdx] > nfeats ? nfeats = lfeats[jdx] : nothing
        end
        vals[i] = lvals
        feats[i] = lfeats
    end

    x = zeros(Float64, nobs, nfeats)
    @inbounds for i in 1:nobs
        @inbounds for jdx in 1:length(feats[i])
            j = feats[i][jdx]
            val = vals[i][jdx]
            x[i, j] = val
        end
    end

    if has_query
        return (x=x, y=y, q=q)
    else
        return (x=x, y=y)
    end
end
```

Data loading can then be performed: 

```
dtrain = read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query=true, aws_config)
deval = read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query=true, aws_config)
dtest = read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query=true, aws_config)
```

## Preprocessing

Preprocessing is minimal since all features are parsed as floats and specific files are provided for the train, eval and test split. 

Since several features are fully missing (contain only 0s) in the training dataset, they will be removed from all datasets.

Then, the features, targets and query will be extracted from the parsed libsvm format. 

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

Now we are ready to train our model. We will first define a model configuration using the [`EvoTreeRegressor`](@ref) model constructor. 
Then, we'll use [`fit_evotree`](@ref) to train a boosted tree model. We'll pass optional `x_eval` and `y_eval` arguments, which enable the usage of early stopping. 

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

## Model evalulation

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

To compute the NDCG over a collection of groups, it is handy to leverage DataFrames' convenient `combine - groupby` functionalities: 

```julia
test_df = DataFrame(p=p_test, y=y_test, q=q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = mean(test_df_agg.ndcg)
@info "ndcg_test MSE" ndcg_test
```
