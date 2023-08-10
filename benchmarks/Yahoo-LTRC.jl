using Revise
using CSV
using DataFrames
using EvoTrees
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

# path = "share/data/yahoo-ltrc/set1.valid.txt"
# raw = S3.get_object(
#     "jeremiedb",
#     path,
#     Dict("response-content-type" => "application/octet-stream");
#     aws_config
# )

function read_libsvm(raw::Vector{UInt8}; has_query=false)
    io = IOBuffer(raw)
    lines = readlines(io)
    # lines = CSV.File(raw, header=false);

    offset = 2 # offset for feature idx: y + query entries
    nobs = length(lines)
    nfeats = 0 # number of features

    y = zeros(Float64, nobs)

    if has_query
        offset = 2 # offset for feature idx: y + query entries
        q = zeros(Int, nobs)
    else
        offset = 2 # offset for feature idx: y
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

function read_libsvm_aws(file::String; has_query=false, aws_config=AWSConfig())
    raw = S3.get_object(
        "jeremiedb",
        file,
        Dict("response-content-type" => "application/octet-stream");
        aws_config
    )
    return read_libsvm(raw; has_query)
end

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

p = [6, 5, 4, 3, 2, 1, 0, -1] .+ 100
y = [3, 2, 3, 0, 1, 2, 3, 2]
ndcg(p, y, 6)

# @time dtrain = read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query=true, aws_config)
@time deval = read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query=true, aws_config)
@time dtest = read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query=true, aws_config)

colsums_eval = map(sum, eachcol(deval[:x]))
colsums_test = map(sum, eachcol(deval[:x]))

sum(colsums_eval .== 0)
sum(colsums_test .== 0)
@assert all((colsums_test .== 0) .== (colsums_test .== 0))
drop_cols = colsums_test .== 0

# x_train = dtrain[:x][:, .!drop_cols]
x_eval = deval[:x][:, .!drop_cols]
x_test = dtest[:x][:, .!drop_cols]

# q_train = dtrain[:q]
q_eval = deval[:q]
q_test = dtest[:q]

#####################################
# mse regression
#####################################

# y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

config = EvoTreeRegressor(
    nrounds=1600,
    loss=:mse,
    eta=0.05,
    nbins=128,
    min_weight=10,
    max_depth=10,
    lambda=0.0,
    gamma=0,
    rowsample=0.5,
    colsample=0.5,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m, logger = fit_evotree(
    config;
    x_train = x_test,
    y_train = y_test,
    x_eval,
    y_eval,
    early_stopping_rounds=100,
    print_every_n=25,
    metric=:mse,
    return_logger=true
);

p_eval = m(x_eval);
eval_df = DataFrame(p = p_eval, y = y_eval, q = q_eval)
eval_df_agg = combine(groupby(eval_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_eval = mean(eval_df_agg.ndcg)

p_test = m(x_test);
test_df = DataFrame(p = p_test, y = y_test, q = q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = mean(test_df_agg.ndcg)

#####################################
# logistic regression
#####################################

# y_train = dtrain[:y] ./ 4
y_eval = deval[:y] ./ 4
y_test = dtest[:y] ./ 4

config = EvoTreeRegressor(
    nrounds=1600,
    loss=:logistic,
    eta=0.05,
    nbins=128,
    min_weight=10,
    max_depth=10,
    lambda=0,
    gamma=0,
    rowsample=0.5,
    colsample=0.5,
)

@time m, logger = fit_evotree(
    config;
    x_train = x_test,
    y_train = y_test,
    x_eval,
    y_eval,
    early_stopping_rounds=100,
    print_every_n=25,
    metric=:logloss,
    return_logger=true
);

# use the original y since NDCG is scale sensitive
# y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

p_eval = m(x_eval);
eval_df = DataFrame(p = p_eval, y = y_eval, q = q_eval)
eval_df_agg = combine(groupby(eval_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_eval = mean(eval_df_agg.ndcg)

p_test = m(x_test);
test_df = DataFrame(p = p_test, y = y_test, q = q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = mean(test_df_agg.ndcg)
