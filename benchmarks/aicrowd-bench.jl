using CSV
using DataFrames
using EvoTrees
using XGBoost
using StatsBase: sample

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

path = "share/data/insurance-aicrowd.csv"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
df = DataFrame(CSV.File(raw))
transform!(df, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

target = "event"
feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "pol_no_claims_discount", "pol_coverage", "pol_duration", "pol_sit_duration"]

pol_cov_dict = Dict{String,Float64}(
    "Min" => 1,
    "Med1" => 2,
    "Med2" => 3,
    "Max" => 4)
pol_cov_map(x) = get(pol_cov_dict, x, 4)
transform!(df, "pol_coverage" => ByRow(pol_cov_map) => "pol_coverage")

setdiff(feats, names(df))

nobs = nrow(df)
id_train = sample(1:nobs, Int(round(0.8 * nobs)), replace=false)

df_train = dropmissing(df[id_train, [feats..., target]])
df_eval = dropmissing(df[Not(id_train), [feats..., target]])

x_train = Matrix{Float32}(df_train[:, feats])
x_eval = Matrix{Float32}(df_eval[:, feats])
y_train = Vector{Float32}(df_train[:, target])
y_eval = Vector{Float32}(df_eval[:, target])

config = EvoTreeRegressor(T=Float32,
    loss=:logistic,
    lambda=0.02,
    gamma=0,
    nbins=32,
    max_depth=5,
    rowsample=0.5,
    colsample=0.8,
    nrounds=400,
    tree_type="oblivious",
    eta=0.05)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m = fit_evotree(config; x_train, y_train, x_eval, y_eval, early_stopping_rounds=50, print_every_n=25, metric=:logloss);
pred_eval_evo = m(x_eval) |> vec;

params_xgb = [
    "objective" => "reg:logistic",
    "booster" => "gbtree",
    "eta" => 0.05,
    "max_depth" => 4,
    "lambda" => 10.0,
    "gamma" => 0.0,
    "subsample" => 0.5,
    "colsample_bytree" => 0.8,
    "tree_method" => "hist",
    "max_bin" => 32,
    "print_every_n" => 5]

nthread = Threads.nthreads()
nthread = 8

nrounds = 400
metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x_train, nrounds, label=y_train, param=params_xgb, metrics=metrics, nthread=nthread, silent=1);
pred_eval_xgb = XGBoost.predict(m_xgb, x_eval)

function logloss(p::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval -= (y[i] * log(p[i]) + (1 - y[i]) * log(1 - p[i]))
    end
    eval /= length(p)
    return eval
end

logloss(pred_eval_evo, y_eval)
logloss(pred_eval_xgb, y_eval)