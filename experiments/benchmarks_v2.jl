using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools
using CUDA

nrounds = 200
nthread = Base.Threads.nthreads()

@info nthread
loss = "logistic"
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

# xgboost aprams
params_xgb = [
    "max_depth" => 5,
    "eta" => 0.05,
    "objective" => loss_xgb,
    "print_every_n" => 5,
    "subsample" => 0.5,
    "colsample_bytree" => 0.5,
    "tree_method" => "hist",
    "max_bin" => 64,
]
metrics = [metric_xgb]

# EvoTrees params
params_evo = EvoTreeRegressor(
    T=Float32,
    loss=loss_evo,
    metric=metric_evo,
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
)

nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
x_train = rand(nobs, num_feat)
y_train = rand(size(x_train, 1))

@info "xgboost train:"
@time m_xgb = xgboost(x_train, nrounds, label=y_train, param=params_xgb, metrics=metrics, nthread=nthread, silent=1);
@btime xgboost($x_train, $nrounds, label=$y_train, param=$params_xgb, metrics=$metrics, nthread=$nthread, silent=1);
@info "xgboost predict:"
@time pred_xgb = XGBoost.predict(m_xgb, x_train);
@btime XGBoost.predict($m_xgb, $x_train);

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train, metric=metric_evo);
@btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, x_train);
@btime EvoTrees.predict($m_evo, $x_train);

CUDA.allowscalar(true)
@info "evotrees train GPU:"
params_evo.device = "gpu"
@time m_evo_gpu = fit_evotree(params_evo; x_train, y_train);
@btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train);
@info "evotrees predict GPU:"
@time pred_evo = EvoTrees.predict(m_evo_gpu, x_train);
@btime EvoTrees.predict($m_evo_gpu, $x_train);

# w_train = ones(length(y_train))
# @time m_evo_gpu = fit_evotree(params_evo, x_train, y_train);
# @time m_evo_gpu = fit_evotree(params_evo, x_train, y_train, w_train);