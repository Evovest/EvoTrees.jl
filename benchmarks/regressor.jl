using CUDA
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools
using Random: seed!

run_evo = true
run_xgb = false
nrounds = 200

loss = :mse
tree_type = "binary"
T = Float64
nthreads = Base.Threads.nthreads()

# device_list = [:cpu, :gpu]
device_list = [:gpu]

# nobs_list = Int.([1e5, 1e6, 1e7])
nobs_list = Int.([1e6])

# nfeats_list = [10, 100]
nfeats_list = [100]

# max_depth_list = [6, 11]
max_depth_list = [6]

for device in device_list
    df = DataFrame()
    for nobs in nobs_list
        for nfeats in nfeats_list
            for max_depth in max_depth_list

                _df = DataFrame(
                    :device => device,
                    :nobs => nobs,
                    :nfeats => nfeats,
                    :max_depth => max_depth)

                @info "device: $device | nobs: $nobs | nfeats: $nfeats | max_depth : $max_depth | nthreads: $nthreads"
                seed!(123)
                x_train = rand(T, nobs, nfeats)
                y_train = rand(T, size(x_train, 1))

                if run_evo
                    @info "EvoTrees"

                    params_evo = EvoTreeRegressor(;
                        loss=loss,
                        nrounds,
                        alpha=0.5,
                        lambda=0.0,
                        gamma=0.0,
                        eta=0.05,
                        max_depth,
                        min_weight=1.0,
                        rowsample=0.5,
                        colsample=0.5,
                        nbins=64,
                        tree_type,
                        rng=123,
                    )

                    if nobs == first(nobs_list) && nfeats == first(nfeats_list) && max_depth == first(max_depth_list)
                        @info "warmup"
                        _m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=loss, device, print_every_n=100)
                        _m_evo(x_train; device)
                    end
                    
                    # Add profiling for GPU performance analysis
                    if device == :gpu
                        println("Profiling GPU training...")
                        CUDA.@profile begin
                            t_train_evo = @elapsed m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=loss, device, print_every_n=100)
                        end
                    else
                    t_train_evo = @elapsed m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric=loss, device, print_every_n=100)
                    end
                    
                    @info "train" t_train_evo
                    t_infer_evo = @elapsed pred_evo = m_evo(x_train; device)
                    @info "predict" t_infer_evo

                    params_evo = EvoTreeRegressor(;
                        loss,
                        nrounds,
                        max_depth,
                        lambda=0.0,
                        gamma=0.0,
                        eta=0.05,
                        min_weight=1.0,
                        rowsample=0.5,
                        colsample=0.5,
                        nbins=64,
                        tree_type,
                        rng=123,
                        device
                    )

                    _df = hcat(_df, DataFrame(
                        :train_evo => t_train_evo,
                        :infer_evo => t_infer_evo)
                    )
                end

                if run_xgb
                    @info "XGBoost"
                    if loss == :mse
                        loss_xgb = "reg:squarederror"
                        metric_xgb = "mae"
                    elseif loss == :logloss
                        loss_xgb = "reg:logistic"
                        metric_xgb = "logloss"
                    end
                    tree_method = device == :gpu ? "gpu_hist" : "hist"

                    params_xgb = Dict(
                        :num_round => nrounds,
                        :max_depth => max_depth - 1,
                        :eta => 0.05,
                        :objective => loss_xgb,
                        :print_every_n => 5,
                        :subsample => 0.5,
                        :colsample_bytree => 0.5,
                        :tree_method => tree_method, # hist/gpu_hist
                        :max_bin => 64,
                    )

                    dtrain = DMatrix(x_train, y_train)
                    watchlist = Dict("train" => DMatrix(x_train, y_train))

                    @info "warmup"
                    if nobs == first(nobs_list) && nfeats == first(nfeats_list) && max_depth == first(max_depth_list)
                        _m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                        XGBoost.predict(_m_xgb, x_train)
                    end
                    t_train_xgb = @elapsed m_xgb = xgboost(dtrain; watchlist, nthread=nthreads, verbosity=0, eval_metric=metric_xgb, params_xgb...)
                    @info "train" t_train_xgb
                    t_infer_xgb = @elapsed pred_xgb = XGBoost.predict(m_xgb, x_train)
                    @info "predict" t_infer_xgb

                    _df = hcat(_df, DataFrame(
                        :train_xgb => t_train_xgb,
                        :infer_xgb => t_infer_xgb)
                    )
                end
                append!(df, _df)
            end
        end
    end
    select!(df, Cols(:device, :nobs, :nfeats, :max_depth, r"train_", r"infer_"))
    path = joinpath(@__DIR__, "results", "regressor-$device.csv")
    CSV.write(path, df)
end

