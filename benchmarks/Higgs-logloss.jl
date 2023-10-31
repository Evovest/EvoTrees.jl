using Revise
using Random
using CSV
using DataFrames
using StatsBase
using Statistics: mean, std
using EvoTrees
using Solage: Connectors
using AWS: AWSCredentials, AWSConfig, @service

@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")
bucket = "jeremiedb"

path = "share/data/higgs/HIGGS.arrow"
df_tot = Connectors.read_arrow_aws(path; bucket="jeremiedb", aws_config)

rename!(df_tot, "Column1" => "y")
feature_names = setdiff(names(df_tot), ["y"])
target_name = "y"

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

transform!(df_tot, feature_names .=> percent_rank .=> feature_names)

dtrain = df_tot[1:end-500_000, :];
deval = df_tot[end-500_000+1:end, :];
dtest = df_tot[end-500_000+1:end, :];

config = EvoTreeRegressor(
    loss=:logloss,
    nrounds=5000,
    eta=0.15,
    nbins=128,
    max_depth=9,
    lambda=1.0,
    gamma=0.0,
    rowsample=0.8,
    colsample=0.8,
    min_weight=1,
    rng=123,
)

device = "gpu"
metric = "logloss"
@time m_evo = fit_evotree(config, dtrain; target_name, fnames=feature_names, deval, metric, device, early_stopping_rounds=200, print_every_n=100);

p_test = m_evo(dtest);
@info extrema(p_test)
logloss_test = mean(-dtest.y .* log.(p_test) .+ (dtest.y .- 1) .* log.(1 .- p_test))
@info "LogLoss - dtest" logloss_test
error_test = 1 - mean(round.(Int, p_test) .== dtest.y)
@info "ERROR - dtest" error_test
# ┌ Info: LogLoss - dtest
# └   logloss_test = 0.4716574579097044
# ┌ Info: ERROR - dtest
# └   error_test = 0.229522

@info "XGBoost"
@info "train"
using XGBoost
params_xgb = Dict(
    :num_round => 4000,
    :max_depth => 8,
    :eta => 0.15,
    :objective => "reg:logistic",
    :print_every_n => 5,
    :gamma => 0,
    :lambda => 1,
    :subsample => 0.8,
    :colsample_bytree => 0.8,
    :tree_method => "gpu_hist", # hist/gpu_hist
    :max_bin => 128,
)

dtrain_xgb = DMatrix(select(dtrain, feature_names), dtrain.y)
watchlist = Dict("eval" => DMatrix(select(deval, feature_names), deval.y));
@time m_xgb = xgboost(dtrain_xgb; watchlist, nthread=Threads.nthreads(), verbosity=0, eval_metric="logloss", params_xgb...);

pred_xgb = XGBoost.predict(m_xgb, DMatrix(select(deval, feature_names)));
@info extrema(pred_xgb)
# (1.9394008f-6, 0.9999975f0)
logloss_test = mean(-dtest.y .* log.(pred_xgb) .+ (dtest.y .- 1) .* log.(1 .- pred_xgb))
@info "LogLoss - dtest" logloss_test
error_test = 1 - mean(round.(Int, pred_xgb) .== dtest.y)
@info "ERROR - xgb test" error_test
# ┌ Info: LogLoss - dtest
# └   logloss_test = 0.4710665675338929
# ┌ Info: ERROR - xgb test
# └   error_test = 0.22987999999999997
