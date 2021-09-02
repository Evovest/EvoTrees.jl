using Statistics
using StatsBase:sample
using XGBoost
using EvoTrees
using BenchmarkTools
using CUDA

nrounds = 200
nthread = 16

# xgboost aprams
params_xgb = ["max_depth" => 5,
         "eta" => 0.05,
         "objective" => "reg:squarederror",
         "print_every_n" => 5,
         "subsample" => 0.5,
         "colsample_bytree" => 0.5,
         "tree_method" => "hist",
         "max_bin" => 64]
metrics = ["rmse"]

# EvoTrees params
params_evo = EvoTreeRegressor(T=Float32,
        loss=:linear, metric=:mse,
        nrounds=nrounds, α=0.5,
        λ=0.0, γ=0.0, η=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=0.5, colsample=0.5, nbins=64)


nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(nobs, num_feat)
Y = rand(size(X, 1))

@info "xgboost train:"
@time m_xgb = xgboost(X, nrounds, label=Y, param=params_xgb, metrics=metrics, nthread=nthread, silent=1);
@btime xgboost($X, $nrounds, label=$Y, param=$params_xgb, metrics=$metrics, nthread=$nthread, silent=1);
@info "xgboost predict:"
@time pred_xgb = XGBoost.predict(m_xgb, X);
@btime XGBoost.predict($m_xgb, $X);

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo, X, Y);
@btime fit_evotree($params_evo, $X, $Y);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, X);
@btime EvoTrees.predict($m_evo, $X);

CUDA.allowscalar(true)
@info "evotrees train GPU:"
params_evo.device = "gpu"
@time m_evo_gpu = fit_evotree(params_evo, X, Y);
@btime fit_evotree($params_evo, $X, $Y);
@info "evotrees predict GPU:"
@time pred_evo = EvoTrees.predict(m_evo_gpu, X);
@btime EvoTrees.predict($m_evo_gpu, $X);