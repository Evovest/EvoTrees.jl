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
tree_type = :binary
T = Float32
nthreads = Base.Threads.nthreads()

# device_list = [:cpu, :gpu]
device_list = [:gpu]

nobs_list = Int.([1e5, 1e6, 1e7])
# nobs_list = Int.([1e5])

nfeats_list = [10, 100]
# nfeats_list = [10]

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
                y_train = rand(T, size(x_train, 1))

                if run_evo
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
                        rng=123,
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
                    tree_method = _device == :gpu ? "gpu_hist" : "hist"

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
    path = joinpath(@__DIR__, "results", "regressor-$_device.csv")
    CSV.write(path, df)
end

