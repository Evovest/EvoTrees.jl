using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools
import CUDA

nobs = Int(1e6)
num_feat = Int(100)
nrounds = 200
num_class = 5
verbosity = 1
T = Float64
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $num_feat features. nthread: $nthread"
x_train = rand(T, nobs, num_feat)
y_train = rand(1:num_class, size(x_train, 1))

loss_xgb = "multi:softmax"
metric_xgb = "mlogloss"
metric_evo = :mlogloss

# xgboost aprams
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
    :num_class => num_class,
)
@info "xgboost train:"
metrics = [metric_xgb]
dtrain = DMatrix(x_train, y_train .- 1);
watchlist = Dict("train" => DMatrix(x_train, y_train .- 1))
@time m_xgb = xgboost(dtrain; watchlist, nthread=nthread, verbosity=0, params_xgb...);
@info "xgboost predict:"
@time pred_xgb = XGBoost.predict(m_xgb, x_train);
# @btime XGBoost.predict($m_xgb, $x_train);

# EvoTrees params
params_evo = EvoTreeClassifier(;
    nrounds=200,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64)

@info "EvoTrees CPU"
device = "cpu"
# @info "train - no eval"
# @time m_evo = fit_evotree(params_evo; x_train, y_train, device, verbosity, print_every_n=100);
# @time m_evo = fit_evotree(params_evo; x_train, y_train, device, verbosity, print_every_n=100);
@info "train - eval"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, print_every_n=100);
@info "evotrees predict CPU:"
@time pred_evo = m_evo(x_train);
@btime m_evo($x_train);

@info "evotrees train GPU:"
device = "gpu"
# @info "train - no eval"
# @time m_evo = fit_evotree(params_evo; x_train, y_train, device, verbosity, print_every_n=100);
# @time m_evo = fit_evotree(params_evo; x_train, y_train, device, verbosity, print_every_n=100);
@info "train - eval"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, print_every_n=100);
# @btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=metric_evo);
@info "evotrees predict GPU:"
@time pred_evo = m_evo(x_train; device);
@btime m_evo($x_train; device);