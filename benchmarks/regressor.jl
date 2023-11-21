using Revise
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using XGBoost
# using LightGBM
using EvoTrees
using BenchmarkTools
using Random: seed!
import CUDA

### v.0.15.1
# desktop | 1e6 | depth 11 | cpu: 37.2s
# desktop | 10e6 | depth 11 | cpu

### v0.16.5
# desktop | 1e6 | depth 11 | cpu: 31s gpu: 50 sec  | xgboost cpu: 26s
# desktop | 10e6 | depth 11 | cpu 200s gpu: 80 sec | xgboost cpu: 267s

#threads
# laptop depth 6: 12.717845 seconds (2.08 M allocations: 466.228 MiB)

# for device in ["cpu", "gpu"]
#     for nobs in Int.([1e5, 1e6, 1e7])
#         for nfeats in [10, 100]
#             for max_depth in [6, 11]

# for device in [device]
#     for nobs in Int.([1e5, 1e6, 1e7])
#         for nfeats in [10, 100]
#             for max_depth in [11]

df = DataFrame()
device = "gpu"
for device in [device]
    for nobs in Int.([1e5, 1e6, 1e7])
        for nfeats in [10, 100]
            for max_depth in [6, 11]

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
                # m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                # t_train_xgb = @elapsed m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                t_train_xgb = missing
                # @btime m_xgb = xgboost($dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric = metric_xgb, params_xgb...);
                @info "predict"
                # pred_xgb = XGBoost.predict(m_xgb, x_train)
                # t_infer_xgb = @elapsed pred_xgb = XGBoost.predict(m_xgb, x_train)
                t_infer_xgb = missing
                # @btime XGBoost.predict($m_xgb, $x_train);

                # @info "lightgbm train:"
                # m_gbm = LGBMRegression(
                #     objective = "regression",
                #     boosting = "gbdt",
                #     num_iterations = 200,
                #     learning_rate = 0.05,
                #     num_leaves = 256,
                #     max_depth = 5,
                #     tree_learner = "serial",
                #     num_threads = Sys.CPU_THREADS,
                #     histogram_pool_size = -1.,
                #     min_data_in_leaf = 1,
                #     min_sum_hessian_in_leaf = 0,
                #     max_delta_step = 0,
                #     min_gain_to_split = 0,
                #     feature_fraction = 0.5,
                #     feature_fraction_seed = 2,
                #     bagging_fraction = 0.5,
                #     bagging_freq = 1,
                #     bagging_seed = 3,
                #     max_bin = 64,
                #     bin_construct_sample_cnt = 200000,
                #     data_random_seed = 1,
                #     is_sparse = false,
                #     min_data_per_group = 1,
                #     metric = ["mae"],
                #     metric_freq = 10,
                #     # early_stopping_round = 10,
                # )
                # @time gbm_results = fit!(m_gbm, x_train, y_train, (x_train, y_train))
                # @time pred_gbm = LightGBM.predict(m_gbm, x_train) |> vec

                _df = DataFrame(
                    :device => device,
                    :nobs => nobs,
                    :nfeats => nfeats,
                    :max_depth => max_depth,
                    :train_evo => t_train_evo,
                    :train_xgb => t_train_xgb,
                    :infer_evo => t_infer_evo,
                    :infer_xgb => t_infer_xgb)
                append!(df, _df)

            end
        end
    end
end

path = joinpath(@__DIR__, "regressor-$device.csv")
CSV.write(path, df)
