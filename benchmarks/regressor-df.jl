using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using DataFrames
using BenchmarkTools
using Random: seed!
import CUDA

nobs = Int(1e6)
num_feat = Int(100)
nrounds = 200
T = Float64
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $num_feat features. nthread: $nthread"
seed!(123)
x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

@info nthread
loss = "mse"
if loss == "mse"
    loss_xgb = "reg:squarederror"
    metric_xgb = "mae"
    loss_evo = :mse
    metric_evo = :mae
elseif loss == "logloss"
    loss_xgb = "reg:logistic"
    metric_xgb = "logloss"
    loss_evo = :logloss
    metric_evo = :logloss
end

@info "XGBoost"
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

# dtrain = DMatrix(x_train, y_train)
# watchlist = Dict("train" => DMatrix(x_train, y_train))
# @time m_xgb = xgboost(dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric=metric_xgb, params_xgb...);
# # @btime m_xgb = xgboost($dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric=metric_xgb, params_xgb...);
# @info "xgboost predict:"
# @time pred_xgb = XGBoost.predict(m_xgb, x_train);
# # @btime XGBoost.predict($m_xgb, $x_train);

@info "EvoTrees"
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
target_name = "y"
verbosity = 0

params_evo = EvoTreeRegressor(
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

@info "EvoTrees CPU"
device = "cpu"

@info "init"
@time m_df, cache_df = EvoTrees.init(params_evo, dtrain; target_name);
@time m_df, cache_df = EvoTrees.init(params_evo, dtrain; target_name);

# @info "train - no eval"
# @time m_evo_df = fit_evotree(params_evo, dtrain; target_name, device, verbosity, print_every_n=100);
# @time m_evo_df = fit_evotree(params_evo, dtrain; target_name, device, verbosity, print_every_n=100);

@info "train - eval"
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
# @time m_evo = fit_evotree(params_evo, dtrain; target_name, device);
# @btime fit_evotree($params_evo, $dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@info "predict"
@time pred_evo = m_evo(dtrain);
@btime m_evo($dtrain);

@info "EvoTrees GPU"
device = "gpu"
@info "train"
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@time m_evo = fit_evotree(params_evo, dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
# @btime m_evo = fit_evotree($params_evo, $dtrain; target_name, device);
# @btime fit_evotree($params_evo, $dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@info "predict"
@time pred_evo = m_evo(dtrain; device);
@btime m_evo($dtrain; device);
