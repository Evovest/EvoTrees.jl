using Revise
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools
using CUDA

nrounds = 200
nobs = Int(1e6)
num_feat = Int(100)
nthread = Base.Threads.nthreads()

# EvoTrees params
params_evo = EvoTreeMLE(
    T=Float64,
    loss=:gaussian,
    nrounds=nrounds,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=100.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
)

@info "testing with: $nobs observations | $num_feat features."
x_train = rand(nobs, num_feat)
y_train = rand(size(x_train, 1))

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=:gaussian, print_every_n=100);
@btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=:gaussian);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, x_train);
@btime EvoTrees.predict($m_evo, $x_train);

CUDA.allowscalar(true)
@info "evotrees train GPU:"
params_evo.device = "gpu"
@time m_evo_gpu = fit_evotree(params_evo; x_train, y_train);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=:gaussian, print_every_n=100);
@btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=:gaussian);
@info "evotrees predict GPU:"
@time pred_evo = EvoTrees.predict(m_evo_gpu, x_train);
@btime EvoTrees.predict($m_evo_gpu, $x_train);


################################
# Logistic
################################
params_evo = EvoTreeMLE(
    T=Float64,
    loss=:logistic,
    nrounds=nrounds,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=100.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
)

@info "testing with: $nobs observations | $num_feat features."
x_train = rand(nobs, num_feat)
y_train = rand(size(x_train, 1))

@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=:logistic_mle, print_every_n=100);
@btime fit_evotree($params_evo; x_train=$x_train, y_train=$y_train, x_eval=$x_train, y_eval=$y_train, metric=:logistic_mle);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, x_train);
@btime EvoTrees.predict($m_evo, $x_train);