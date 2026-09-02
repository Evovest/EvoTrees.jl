using PythonCall
using CSV
using DataFrames
using Statistics
using EvoTrees
using Random: seed!

xgb = pyimport("xgboost")
np = pyimport("numpy")

nrounds = 200
T = Float32
nthreads = Base.Threads.nthreads()

# Ranking cost grows with the number of documents per query, so group size is swept alongside
# the usual shapes. XGBoost truncates pairs by default, EvoTrees scores every pair.
group_size_list = [10, 20, 100]
nobs_list = Int.([1e5, 1e6])
nfeats_list = [100]
max_depth_list = [6]
ndcg_k = 10

device_list = [:cpu, :gpu]

for _device in device_list
    df = DataFrame()
    for nobs in nobs_list
        for nfeats in nfeats_list
            for max_depth in max_depth_list
                for group_size in group_size_list

                    @info "device: $_device | nobs: $nobs | nfeats: $nfeats | max_depth: $max_depth | group_size: $group_size | nthreads: $nthreads"
                    seed!(123)
                    x_train = rand(T, nobs, nfeats)
                    y_train = T.(rand(0:4, nobs))
                    ngroups = nobs ÷ group_size
                    q_train = UInt32.(repeat(1:ngroups, inner=group_size))

                    _df = DataFrame(
                        :device => _device,
                        :nobs => nobs,
                        :nfeats => nfeats,
                        :max_depth => max_depth,
                        :group_size => group_size)

                    # EvoTrees, mse against lambdarank on the same groups
                    for (name, loss, metric) in ((:mse, :mse, :mae), (:lambdarank, :lambdarank, :ndcg))
                        params_evo = EvoTreeRegressor(;
                            loss, metric, nrounds, max_depth, ndcg_k,
                            eta=0.05, min_weight=1.0, rowsample=0.5, colsample=0.5,
                            nbins=64, tree_type=:binary, seed=123, device=_device)
                        EvoTrees.fit(params_evo; x_train, y_train, group_train=q_train,
                            x_eval=x_train, y_eval=y_train, group_eval=q_train, print_every_n=1000)
                        t = @elapsed EvoTrees.fit(params_evo; x_train, y_train, group_train=q_train,
                            x_eval=x_train, y_eval=y_train, group_eval=q_train, print_every_n=1000)
                        _df[!, Symbol("train_evo_", name)] = [t]
                    end

                    # XGBoost, reg:squarederror against rank:ndcg on the same groups
                    x_np = np.array(x_train)
                    y_np = np.array(y_train)
                    group_np = np.array(fill(group_size, ngroups))
                    _dev = _device == :cpu ? "cpu" : "cuda"

                    for (name, objective, metric_xgb) in
                        ((:mse, "reg:squarederror", "mae"), (:ndcg, "rank:ndcg", "ndcg"))

                        dtrain = xgb.DMatrix(x_np, label=y_np)
                        objective == "rank:ndcg" && dtrain.set_group(group_np)
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
                            "lambdarank_num_pair_per_sample" => ndcg_k,
                        ) |> pydict

                        xgb.train(params, dtrain, num_boost_round=5,
                            evals=pylist([(dtrain, "train")]), verbose_eval=1000)
                        t = @elapsed xgb.train(params, dtrain, num_boost_round=nrounds,
                            evals=pylist([(dtrain, "train")]), verbose_eval=1000)
                        _df[!, Symbol("train_xgb_", name)] = [t]
                    end

                    _df[!, :evo_ndcg_over_mse] = _df.train_evo_lambdarank ./ _df.train_evo_mse
                    _df[!, :xgb_ndcg_over_mse] = _df.train_xgb_ndcg ./ _df.train_xgb_mse
                    append!(df, _df)
                end
            end
        end
    end
    path = joinpath(@__DIR__, "results", "ranking-$_device.csv")
    CSV.write(path, df)
end
