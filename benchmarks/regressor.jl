using Revise
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using XGBoost
using EvoTrees
using BenchmarkTools
using Random: seed!
# import CUDA

### v.0.15.1
# desktop | 1e6 | depth 11 | cpu: 37.2s
# desktop | 10e6 | depth 11 | cpu

### v0.16.5
# desktop | 1e6 | depth 11 | cpu: 31s gpu: 50 sec  | xgboost cpu: 26s
# desktop | 10e6 | depth 11 | cpu 200s gpu: 80 sec | xgboost cpu: 267s

#threads
# laptop depth 6: 12.717845 seconds (2.08 M allocations: 466.228 MiB)

run_evo = true
run_xgb = false

# device_list = ["cpu", "gpu"]
device_list = ["cpu"]

# nobs_list = Int.([1e5, 1e6, 1e7])
nobs_list = Int.([1e6])

# nfeats_list = [10, 100, 1000]
nfeats_list = [100]

# max_depth_list in [6, 11]
max_depth_list = [11]


df = DataFrame()
for device in device_list
    for nobs in nobs_list
        for nfeats in nfeats_list
            for max_depth in max_depth_list

                _df = DataFrame(
                    :device => device,
                    :nobs => nobs,
                    :nfeats => nfeats,
                    :max_depth => max_depth)

                max_nrounds = 200
                tree_type = "binary"
                T = Float64
                nthreads = Base.Threads.nthreads()
                @info "device: $device | nobs: $nobs | nfeats: $nfeats | max_depth : $max_depth | nthreads: $nthreads | tree_type : $tree_type"
                seed!(123)
                x_train = rand(T, nobs, nfeats)
                y_train = rand(T, size(x_train, 1))

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
                tree_method = device == "gpu" ? "gpu_hist" : "hist"

                if run_evo
                    @info "EvoTrees"
                    verbosity = 1

                    params_evo = EvoTreeRegressor(;
                        loss=loss_evo,
                        nrounds=max_nrounds,
                        alpha=0.5,
                        lambda=0.0,
                        gamma=0.0,
                        eta=0.05,
                        max_depth=max_depth,
                        min_weight=1.0,
                        rowsample=0.5,
                        colsample=0.5,
                        nbins=64,
                        tree_type,
                        rng=123
                    )

                    @info "train - eval"
                    @time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, verbosity, print_every_n=100)
                    t_train_evo = @elapsed m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=metric_evo, device, verbosity, print_every_n=100)

                    @info "predict"
                    @time pred_evo = m_evo(x_train; device)
                    t_infer_evo = @elapsed pred_evo = m_evo(x_train; device)

                    _df = hcat(_df, DataFrame(
                        :train_evo => t_train_evo,
                        :infer_evo => t_infer_evo)
                    )
                end

                if run_xgb
                    @info "XGBoost"
                    params_xgb = Dict(
                        :num_round => max_nrounds,
                        :max_depth => max_depth - 1,
                        :eta => 0.05,
                        :objective => loss_xgb,
                        :print_every_n => 5,
                        :subsample => 0.5,
                        :colsample_bytree => 0.5,
                        :tree_method => tree_method, # hist/gpu_hist
                        :max_bin => 64,
                    )

                    @info "train"
                    dtrain = DMatrix(x_train, y_train)
                    watchlist = Dict("train" => DMatrix(x_train, y_train))
                    m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                    t_train_xgb = @elapsed m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                    @info "predict"
                    pred_xgb = XGBoost.predict(m_xgb, x_train)
                    t_infer_xgb = @elapsed pred_xgb = XGBoost.predict(m_xgb, x_train)

                    _df = hcat(_df, DataFrame(
                        :train_xgb => t_train_xgb,
                        :infer_xgb => t_infer_xgb)
                    )
                end
                append!(df, _df)
            end
        end
    end
end

path = joinpath(@__DIR__, "regressor-$device.csv")
CSV.write(path, df)
