using CUDA
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools
using Random: seed!
# using XGBoost

run_evo = true
nrounds = 200
n_targets = 1

loss = :mse
tree_type = :binary
T = Float32
nthreads = Base.Threads.nthreads()

device_list = [:cpu, :gpu]
# device_list = [:gpu]

nobs_list = Int.([1e5, 1e6, 1e7])
# nobs_list = Int.([1e6])

nfeats_list = [10, 100]
# nfeats_list = [100]

max_depth_list = [6, 11]
# max_depth_list = [6]

# nobs = first(nobs_list)
# nfeats = first(nfeats_list)
# max_depth = first(max_depth_list)
# _device = first(device_list)

for _device in device_list
    df = DataFrame()
    for nobs in nobs_list
        for nfeats in nfeats_list
            for max_depth in max_depth_list

                _df = DataFrame(
                    :device => _device,
                    :nobs => nobs,
                    :nfeats => nfeats,
                    :max_depth => max_depth)

                @info "device: $_device | nobs: $nobs | nfeats: $nfeats | max_depth : $max_depth | nthreads: $nthreads"
                seed!(123)
                x_train = rand(T, nobs, nfeats)
                y_train = n_targets == 1 ? rand(T, size(x_train, 1)) : rand(T, size(x_train, 1), n_targets)

                @info "EvoTrees"
                loss == :mse ? metric = :mae : metric = loss

                params_evo = EvoTreeRegressor(;
                    loss,
                    metric,
                    nrounds,
                    max_depth,
                    eta=0.05,
                    min_weight=1.0,
                    rowsample=0.5,
                    colsample=0.5,
                    nbins=64,
                    tree_type,
                    seed=123,
                    device=_device
                )

                if nobs == first(nobs_list) && nfeats == first(nfeats_list) && max_depth == first(max_depth_list)
                    @info "warmup"
                    _m_evo = EvoTrees.fit(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100)
                    _m_evo(x_train; device=_device)
                end
                t_train_evo = @elapsed m_evo = EvoTrees.fit(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, print_every_n=100)
                @info "train" t_train_evo
                t_infer_evo = @elapsed pred_evo = m_evo(x_train; device=_device)
                @info "predict" t_infer_evo

                _df = hcat(_df, DataFrame(
                    :train_evo => t_train_evo,
                    :infer_evo => t_infer_evo)
                )
                append!(df, _df)
            end
        end
    end
    select!(df, Cols(:device, :nobs, :nfeats, :max_depth, r"train_", r"infer_"))
    path = joinpath(@__DIR__, "results", "regressor-$_device.csv")
    CSV.write(path, df)
end
