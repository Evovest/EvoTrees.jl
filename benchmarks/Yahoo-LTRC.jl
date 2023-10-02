using Revise
using CSV
using DataFrames
using EvoTrees
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!
using ReadLIBSVM
# using GLMakie

# data is C14 - Yahoo! Learning to Rank Challenge
# data can be obtained though a request to https://webscope.sandbox.yahoo.com/
using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

function read_libsvm_aws(file::String; has_query=false, aws_config=AWSConfig())
    raw = S3.get_object("jeremiedb", file, Dict("response-content-type" => "application/octet-stream"); aws_config)
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

@time dtrain = read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query=true, aws_config)
@time deval = read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query=true, aws_config)
@time dtest = read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query=true, aws_config)

colsums_train = map(sum, eachcol(dtrain[:x]))
# colsums_eval = map(sum, eachcol(deval[:x]))
colsums_test = map(sum, eachcol(deval[:x]))

sum(colsums_train .== 0)
sum(colsums_test .== 0)
@assert all((colsums_train .== 0) .== (colsums_test .== 0))
drop_cols = colsums_train .== 0

x_train = dtrain[:x][:, .!drop_cols]
x_eval = deval[:x][:, .!drop_cols]
x_test = dtest[:x][:, .!drop_cols]

# x_train_miss = x_train .== 0
# x_eval_miss = x_eval .== 0
# x_test_miss = x_test .== 0

# x_train[x_train.==0] .= 0.5
# x_eval[x_eval.==0] .= 0.5
# x_test[x_test.==0] .= 0.5

# x_train = hcat(x_train, x_train_miss)
# x_eval = hcat(x_eval, x_eval_miss)
# x_test = hcat(x_test, x_test_miss)

q_train = dtrain[:q]
q_eval = deval[:q]
q_test = dtest[:q]

y_train = dtrain[:y];
y_eval = deval[:y];
y_test = dtest[:y];

#####################################
# mse regression
#####################################

y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

config = EvoTreeRegressor(
    nrounds=6000,
    loss=:mse,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m_mse, logger_mse = fit_evotree(
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
test_df = DataFrame(p=p_test, y=y_test, q=q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = round(mean(test_df_agg.ndcg), sigdigits=5)
@info "MSE - test data - MSE model" mean((p_test .- y_test) .^ 2)
@info "NDCG - test data - MSE model" ndcg_test

#####################################
# logistic regression
#####################################
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

@time m_logloss, logger_logloss = fit_evotree(
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

# use the original y since NDCG is scale sensitive
y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

# p_eval = m(x_eval);
# eval_df = DataFrame(p = p_eval, y = y_eval, q = q_eval)
# eval_df_agg = combine(groupby(eval_df, "q"), ["p", "y"] => ndcg => "ndcg")
# ndcg_eval = mean(eval_df_agg.ndcg)

p_test = m_logloss(x_test);
test_df = DataFrame(p=p_test, y=y_test, q=q_test)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = round(mean(test_df_agg.ndcg), sigdigits=5)
@info "NDCG - test data - LogLoss model" ndcg_test

#####################################
# logistic regression on DataFrame
#####################################
target_name = "y"

df_train = DataFrame(x_train, :auto)
df_train.y = dtrain[:y] ./ 4
df_train.q = dtrain[:q]

df_eval = DataFrame(x_eval, :auto)
df_eval.y = deval[:y] ./ 4
df_eval.q = deval[:q]

df_test = DataFrame(x_test, :auto)
df_test.y = dtest[:y] ./ 4
df_test.q = dtest[:q]

function rank_target_norm(y::AbstractVector)
    out = similar(y)
    if minimum(y) == maximum(y)
        out .= 0.5
    else
        out .= (y .- minimum(y)) ./ (maximum(y) - minimum(y))
    end
    return out
end

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

feature_names_raw = setdiff(names(df_train), ["y", "q"])
feature_names_rel = feature_names_raw .* "_rel"

transform!(df_train, feature_names_raw .=> percent_rank .=> feature_names_rel)
transform!(df_eval, feature_names_raw .=> percent_rank .=> feature_names_rel)
transform!(df_test, feature_names_raw .=> percent_rank .=> feature_names_rel)

feature_names = setdiff(names(df_train), ["y", "q"])

# df_train = transform!(
#     groupby(df_train, "q"),
#     "y" => rank_target_norm => "y")

# df_eval = transform!(
#     groupby(df_eval, "q"),
#     "y" => rank_target_norm => "y")

# df_test = transform!(
#     groupby(df_test, "q"),
#     "y" => rank_target_norm => "y")

minimum(df_eval.y)
maximum(df_eval.y)

config = EvoTreeRegressor(
    nrounds=6000,
    loss=:logloss,
    eta=0.01,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

@time m_logloss_df, logger_logloss_df = fit_evotree(
    config,
    df_train;
    target_name,
    fnames=feature_names_raw,
    deval=df_eval,
    early_stopping_rounds=200,
    print_every_n=50,
    metric=:logloss,
    return_logger=true
);

m_logloss_df.info
p_test_df = m_logloss_df(df_test);
# p_test_mat = m_logloss_df(x_test);

EvoTrees.importance(m_logloss_df)

p_test = m_logloss_df(df_test);
test_df = DataFrame(p=p_test, y=dtest[:y], q=dtest[:q])
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = mean(test_df_agg.ndcg)
# ndcg_test = 0.8022558972243291
# ndcg_test = 0.8020754563069513
@info "NDCG - test data - LogLoss DF model" ndcg_test
