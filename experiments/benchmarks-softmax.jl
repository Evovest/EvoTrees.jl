using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools
using CUDA
using Random

nrounds = 20
num_class = 5
nthread = Base.Threads.nthreads()

@info nthread
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
metrics = [metric_xgb]

# EvoTrees params
params_evo = EvoTreeClassifier(;
    T=Float32,
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=9,
    min_weight=1.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64)

nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
x_train = rand(nobs, num_feat)
y_train = rand(1:num_class, size(x_train, 1))

@info "xgboost train:"
dtrain = DMatrix(x_train, y_train .- 1)
watchlist = Dict("train" => DMatrix(x_train, y_train .- 1))
@time m_xgb = xgboost(dtrain; watchlist, nthread=nthread, verbosity=0, params_xgb...);
@info "xgboost predict:"
@time pred_xgb = XGBoost.predict(m_xgb, x_train);
@btime XGBoost.predict($m_xgb, $x_train);

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, print_every_n=100);
# @btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=metric_evo);
@time fit_evotree(params_evo; x_train, y_train);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, x_train);
@btime EvoTrees.predict($m_evo, $x_train);

@info "evotrees train GPU:"
params_evo.device = "gpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, print_every_n=100);
# @btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=metric_evo);
@info "evotrees predict GPU:"
@time pred_evo = EvoTrees.predict(m_evo_gpu, x_train);
@btime EvoTrees.predict($m_evo_gpu, $x_train);