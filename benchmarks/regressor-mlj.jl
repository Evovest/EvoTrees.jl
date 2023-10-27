using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using DataFrames
using BenchmarkTools
using Random: seed!
import CUDA
using MLJ

nobs = Int(2e6)
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
    loss_evo = :mse
    metric_evo = :mae
elseif loss == "logloss"
    loss_evo = :logloss
    metric_evo = :logloss
end

@info "EvoTrees"
dtrain = DataFrame(x_train, :auto)
# dtrain.y .= y_train
# target_name = "y"
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

iterated_model = IteratedModel(
    model=params_evo,
    resampling=Holdout(; fraction_train=0.5),
    measures=rmse,
    controls=[Step(5),
        Patience(200),
        NumberLimit(40)],
    retrain=false)

mach = machine(iterated_model, dtrain, y_train)
@time fit!(mach);

@info "init"
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


using MLJBase
using MLJModels
using Tables

EvoTreeBooster = @load EvoTreeRegressor
booster = EvoTreeBooster()

X, y = make_regression(1000, 5)

# this works:
mach = machine(booster, X, y) |> fit!

# this doesn't
X, y = make_regression(1_000_000, 100);
@time X = DataFrame(X);
@time X = Tables.rowtable(X);
@time X = Tables.columntable(X);

mach = machine(booster, X, y) |> fit!

schema = Tables.schema(dtrain)
schema.names