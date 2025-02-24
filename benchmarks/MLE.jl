using Revise
using Statistics
using StatsBase: sample
using EvoTrees
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

@info "Gaussian MLE"
params_evo = EvoTreeMLE(
    loss=:gaussian_mle,
    nrounds=200,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=100.0,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
)

@info "evotrees train CPU:"
params_evo.device = :cpu
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@info "evotrees predict CPU:"
@time pred_evo = m_evo(x_train);
@btime m_evo($x_train);

CUDA.allowscalar(true)
@info "evotrees train GPU:"
params_evo.device = :cpu
# @time m_evo = fit_evotree(params_evo; x_train, y_train);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@info "evotrees predict GPU:"
@time pred_evo = m_evo(x_train; device);
@btime m_evo($x_train; device);

################################
# Logistic
################################
@info "Logistic MLE"
params_evo = EvoTreeMLE(
    loss=:logistic_mle,
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
params_evo.device = :cpu
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100);
@info "evotrees predict CPU:"
@time pred_evo = m_evo(x_train);
@btime m_evo($x_train);