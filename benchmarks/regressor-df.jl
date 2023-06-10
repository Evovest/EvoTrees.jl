using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using DataFrames
using BenchmarkTools
import CUDA

nrounds = 200
nobs = Int(1e6)
num_feat = Int(100)
T = Float32
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $num_feat features."
x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

@info nthread
loss = "linear"
if loss == "linear"
    loss_xgb = "reg:squarederror"
    metric_xgb = "mae"
    loss_evo = :linear
    metric_evo = :mae
elseif loss == "logistic"
    loss_xgb = "reg:logistic"
    metric_xgb = "logloss"
    loss_evo = :logistic
    metric_evo = :logloss
end

@info "xgboost train:"
params_xgb = Dict(
    :num_round => nrounds,
    :max_depth => 5,
    :eta => 0.05,
    :objective => loss_xgb,
    :print_every_n => 5,
    :subsample => 0.5,
    :colsample_bytree => 0.5,
    :tree_method => "hist",
    :max_bin => 64,
)

dtrain = DMatrix(x_train, y_train .- 1)
watchlist = Dict("train" => DMatrix(x_train, y_train .- 1))
@time m_xgb = xgboost(dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric=metric_xgb, params_xgb...);
# @btime m_xgb = xgboost($dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric=metric_xgb, params_xgb...);
@info "xgboost predict:"
@time pred_xgb = XGBoost.predict(m_xgb, x_train);
# @btime XGBoost.predict($m_xgb, $x_train);

@info "EvoTrees"
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
target_name = "y"
verbosity = 0

params_evo = EvoTreeRegressor(
    T=T,
    loss=loss_evo,
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
    rng=123,
)

@info "evotrees train CPU:"
device = "cpu"
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
# @time m_evo = fit_evotree(params_evo, dtrain; target_name, device);
# @btime fit_evotree($params_evo, $dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@info "evotrees predict CPU:"
@time pred_evo = m_evo(dtrain);
@btime m_evo($dtrain);

@info "evotrees train GPU:"
device = "gpu"
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
# @btime m_evo = fit_evotree($params_evo, $dtrain; target_name, device);
# @btime fit_evotree($params_evo, $dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@info "evotrees predict CPU:"
@time pred_evo = m_evo(dtrain);
@btime m_evo($dtrain);
