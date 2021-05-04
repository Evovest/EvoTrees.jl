using Statistics
using StatsBase:sample
using Revise
using EvoTrees
using MemoryConstrainedTreeBoosting

nrounds = 100

# EvoTrees params
params_evo = EvoTreeRegressor(T=Float32,
        loss=:logistic, metric=:logloss,
        nrounds=nrounds,
        λ=0.5, γ=0.0, η=0.05,
        max_depth=6, min_weight=1.0,
        rowsample=1.0, colsample=0.5, nbins=64)

# MemoryConstrainedTreeBoosting params
params_mctb = (
        weights                 = nothing,
        bin_count               = 128,
        iteration_count         = nrounds,
        min_data_weight_in_leaf = 1.0,
        l2_regularization       = 0.0,
        max_leaves              = 64,
        max_depth               = 6,
        max_delta_score         = 1.0e10, # Before shrinkage.
        learning_rate           = 0.05,
        feature_fraction        = 0.5, # Per tree.
        bagging_temperature     = 0.1,
      )

nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(Float32, nobs, num_feat)
Y = Float32.(rand(Bool, size(X, 1)))


@info "evotrees train CPU:"
params_evo.device = "cpu"
@time m, cache = EvoTrees.init_evotree(params_evo, X, Y);
@time EvoTrees.grow_evotree!(m, cache);
@time m, cache = EvoTrees.init_evotree(params_evo, X, Y);
@time EvoTrees.grow_evotree!(m, cache);
@time m_evo = fit_evotree(params_evo, X, Y);
@time fit_evotree(params_evo, X, Y);
@info "evotrees predict CPU:"
@time pred_evo = EvoTrees.predict(m_evo, X);
@time EvoTrees.predict(m_evo, X);

@info "evotrees train GPU:"
params_evo.device = "gpu"
@time m_evo = fit_evotree(params_evo, X, Y);
@time fit_evotree(params_evo, X, Y);
@info "evotrees predict GPU:"
@time pred_evo = EvoTrees.predict(m_evo, X);
@time EvoTrees.predict(m_evo, X);

@info "MemoryConstrainedTreeBoosting train CPU:"
@time bin_splits, trees = MemoryConstrainedTreeBoosting.train(X, Y; params_mctb...);
@time MemoryConstrainedTreeBoosting.train(X, Y; params_mctb...);
@info "MemoryConstrainedTreeBoosting predict CPU, JITed:"
save_path = tempname()
MemoryConstrainedTreeBoosting.save(save_path, bin_splits, trees)
unbinned_predict = MemoryConstrainedTreeBoosting.load_unbinned_predictor(save_path)
@time pred_mctb = unbinned_predict(X)
@time unbinned_predict(X)