using CSV
using DataFrames
using EvoTrees
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!
using ReadLIBSVM

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

# Test-set NDCG through the package's own metric, so the reported figure is the same
# quantity early stopping selects on rather than a second implementation.
function ndcg_test(m, x, y, q; k=10)
    p = EvoTrees.predict(m, x)
    gi = EvoTrees.build_group_index(q, length(q), "q")
    w = ones(Float32, length(y))
    EvoTrees.ndcg(reshape(Float32.(p), 1, :), Float32.(y), w, Float32[]; group=gi, ndcg_k=k)
end

@time dtrain = read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query=true, aws_config)
@time deval = read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query=true, aws_config)
@time dtest = read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query=true, aws_config)

colsums_train = map(sum, eachcol(dtrain[:x]))
colsums_test = map(sum, eachcol(deval[:x]))

@assert all((colsums_train .== 0) .== (colsums_test .== 0))
drop_cols = colsums_train .== 0

x_train = dtrain[:x][:, .!drop_cols]
x_eval = deval[:x][:, .!drop_cols]
x_test = dtest[:x][:, .!drop_cols]

q_train = dtrain[:q]
q_eval = deval[:q]
q_test = dtest[:q]

y_train = dtrain[:y]
y_eval = deval[:y]
y_test = dtest[:y]

const NDCG_K = 10

#####################################
# mse regression
#####################################
config = EvoTreeRegressor(
    nrounds=6000,
    loss=:mse,
    metric=:ndcg,
    ndcg_k=NDCG_K,
    early_stopping_rounds=200,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

@time m_mse = EvoTrees.fit(
    config;
    x_train, y_train, group_train=q_train,
    x_eval, y_eval, group_eval=q_eval,
    print_every_n=50,
)

p_test = EvoTrees.predict(m_mse, x_test)
@info "MSE - test data - MSE model" mean((p_test .- y_test) .^ 2)
@info "NDCG - test data - MSE model" round(ndcg_test(m_mse, x_test, y_test, q_test; k=NDCG_K), sigdigits=5)

#####################################
# logistic regression
#####################################
# `:logloss` needs the target in [0, 1], so the NDCG tracked during fitting is computed on
# the scaled relevance. Ranking is unaffected, but the logged value is not on the same scale
# as the test figures below, which all use the raw 0-4 relevance.
max_rank = 4
config = EvoTreeRegressor(
    nrounds=6000,
    loss=:logloss,
    metric=:ndcg,
    ndcg_k=NDCG_K,
    early_stopping_rounds=200,
    eta=0.01,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

@time m_logloss = EvoTrees.fit(
    config;
    x_train, y_train=y_train ./ max_rank, group_train=q_train,
    x_eval, y_eval=y_eval ./ max_rank, group_eval=q_eval,
    print_every_n=50,
)

@info "NDCG - test data - LogLoss model" round(ndcg_test(m_logloss, x_test, y_test, q_test; k=NDCG_K), sigdigits=5)

#####################################
# lambdarank
#####################################
config = EvoTreeRegressor(
    nrounds=6000,
    loss=:lambdarank,
    metric=:ndcg,
    ndcg_k=NDCG_K,
    early_stopping_rounds=200,
    eta=0.02,
    nbins=64,
    max_depth=11,
    rowsample=0.9,
    colsample=0.9,
)

@time m_lambda = EvoTrees.fit(
    config;
    x_train, y_train, group_train=q_train,
    x_eval, y_eval, group_eval=q_eval,
    print_every_n=50,
)

@info "NDCG - test data - LambdaRank model" round(ndcg_test(m_lambda, x_test, y_test, q_test; k=NDCG_K), sigdigits=5)

#####################################
# summary
#####################################
for (name, m) in (("mse", m_mse), ("logloss", m_logloss), ("lambdarank", m_lambda))
    @info name trees = length(m.trees) - 1 best_iter = m.info[:logger][:best_iter] ndcg = round(
        ndcg_test(m, x_test, y_test, q_test; k=NDCG_K), sigdigits=5)
end
