using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools
using CUDA

nrounds = 200
nthread = Base.Threads.nthreads()

@info nthread

# EvoTrees params
params_evo = EvoTreeRegressor(
    T=Float32,
    loss="linear",
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=1.0,
    colsample=1.0,
    nbins=64,
    device = "gpu"
)

nobs = Int(11_664_400)
num_feat = Int(36)
@info "testing with: $nobs observations | $num_feat features."
x_train = rand(nobs, num_feat)
y_train = rand(size(x_train, 1))

@info "evotrees train GPU:"
@time m_evo_gpu = fit_evotree(params_evo; x_train, y_train);
for i in 1:5
    @time m_evo_gpu = fit_evotree(params_evo; x_train, y_train);
end
# @time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, print_every_n=100);