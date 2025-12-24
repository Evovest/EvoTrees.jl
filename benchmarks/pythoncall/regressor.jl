using PythonCall
using CSV
using DataFrames
using Statistics
using BenchmarkTools
using StatsBase: sample
using Random: seed!

# Import Python modules
xgb = pyimport("xgboost")
np = pyimport("numpy")

# Parameters
nrounds = 200
loss = :mse
nthreads = Base.Threads.nthreads()

device_list = ["cpu", "gpu"]
# device_list = ["gpu"]

nobs_list = Int.([1e5, 1e6, 1e7])
# nobs_list = Int.([1e7])

nfeats_list = [10, 100]
# nfeats_list = [10]

max_depth_list = [5, 10]
# max_depth_list = [10]

nobs = first(nobs_list)
nfeats = first(nfeats_list)
max_depth = first(max_depth_list)
_device = first(device_list)

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
                # Generate random data
                x_train = rand(Float32, nobs, nfeats)
                y_train = rand(Float32, nobs)

                # Convert to numpy arrays
                x_train_np = np.array(x_train)
                y_train_np = np.array(y_train)

                # Create DMatrix
                dtrain = xgb.DMatrix(x_train_np, label=y_train_np)

                # Set XGBoost parameters
                if loss == :mse
                    objective = "reg:squarederror"
                    metric_xgb = "mae"
                elseif loss == :logloss
                    objective = "reg:logistic"
                    metric_xgb = "logloss"
                end
                _dev = _device == "cpu" ? "cpu" : "cuda"

                params = Dict(
                    "objective" => objective,
                    "eval_metric" => pylist([metric_xgb]),
                    "max_depth" => max_depth,
                    "eta" => 0.05,
                    "tree_method" => "hist",
                    "nthread" => nthreads,
                    "verbosity" => 0,
                    "subsample" => 0.5,
                    "colsample_bytree" => 0.5,
                    "max_bin" => 64,
                    "device" => _dev,
                ) |> pydict

                if nobs == first(nobs_list) && nfeats == first(nfeats_list) && max_depth == first(max_depth_list)
                    @info "warmup"
                    booster = xgb.train(params, dtrain, num_boost_round=5, evals=pylist([(dtrain, "train")]), verbose_eval=100)
                    preds = booster.predict(dtrain)
                end

                train_time = @elapsed booster = xgb.train(params, dtrain, num_boost_round=nrounds, evals=pylist([(dtrain, "train")]), verbose_eval=100)
                @info "train" train_time

                pred_time = @elapsed preds = booster.predict(dtrain)
                @info "predict" pred_time
                # mean(preds)
                # mean(pyconvert(Array, preds))

                _df = hcat(_df, DataFrame(
                    :train_xgb => train_time,
                    :infer_xgb => pred_time)
                )
                append!(df, _df)
            end
        end
    end
    select!(df, Cols(:device, :nobs, :nfeats, :max_depth, r"train_", r"infer_"))
    path = joinpath(@__DIR__, "results", "regressor-xgb-$_device.csv")
    CSV.write(path, df)
end
