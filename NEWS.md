# NEWS

## v0.20.0

### Model structure
- The intercept is stored on `EvoTree` as `bias::Vector{Float32}` (unconstrained / link space). `trees` holds only boosting rounds, so `length(m.trees) == nrounds * bagging_size` (and is empty when `nrounds = 0`).
- `predict` always applies `bias`, then the first `ntree_limit` trees. `ntree_limit=0` is bias only.
- `treeplot` defaults to `n=1` (the first learned tree).
- `max_depth` is the maximum number of splits on a leaf path (`1` is a split with 2 leaves).

### Plotting
- Tree plots now use a **Makie recipe** instead of Plots.jl / RecipesBase. Load a backend (`CairoMakie` or `GLMakie`) and call `treeplot(model)` or `plot(model, 1)`.
- `RecipesBase` and `NetworkLayout` are no longer dependencies. Layout is computed in EvoTrees; the recipe is a package extension loaded with Makie.
- Split labels show the comparison (`≤` for numeric/ordered features, `=` for unordered categoricals). Multi-output leaves show every predicted parameter.

## v0.18.8

Saved `:gaussian_mle` / `:logistic_mle` models from ≤0.18.7 remain compatible: the unconstrained scale is still `exp(x)`.

### Behavioral changes
- **Multiclass Hessian.** `:mlogloss` now uses the softmax diagonal `p(1-p)`. Retraining a classifier can yield different trees; already-saved models still predict as before.
- **Invalid input now errors** instead of logging and continuing. In particular: a single-level classification target, a non-positive target under `:gamma`, an empty row subsample, and `L2` / `bagging_size` / `early_stopping_rounds` outside their valid ranges.
- **Classification eval labels** are encoded against the training levels. `y_eval` must only contain classes seen in `y_train`; a different level order no longer silently scores the wrong prediction columns.
- **MLJ `update`** continues an existing fit only when `nrounds` is the sole hyper-parameter change (and is not reduced). Any other change triggers a full refit.

### MLE losses: Fisher information
- `:gaussian_mle` and `:logistic_mle` now take Newton steps with the **Fisher information** (expected Hessian) rather than the observed Hessian. The Fisher matrix is diagonal and positive definite for these location-scale families, which stabilizes split finding and leaf weights.
- The unconstrained scale parameter remains `exp(x)`. `predict` still returns the positive scale (`σ` / `s`). User-facing offsets remain the positive scale and are mapped internally with `log`.

### Multi-target regression
- Several losses can now be fit jointly on a vector of targets, with shared tree structure and a vector-valued leaf. Pass a matrix `y_train` of size `(nobs, n_targets)`, matching `x_train`'s `(nobs, nfeats)`, or `target_name=["y1", "y2", ...]` on a table.
- Supported: `:mse`, `:logloss`, `:poisson`, `:gamma`, `:tweedie`, `:mae`, `:quantile`, `:cred_std`, `:cred_var`, `:gaussian_mle`, `:logistic_mle`. Not supported: `:mlogloss`, `:multiquantile`.
- Predictions are `(nobs, n_targets)`. MLE losses return `(nobs, 2 * n_targets)` with interleaved location and positive scale per target.

```julia
config = EvoTreeRegressor(; loss=:mse, nrounds=200)
m = fit(config, dtrain; target_name=["y", "y2"])
pred = m(dtrain)  # size (nobs, 2)
```

### GPU: KernelAbstractions, AMD and Metal
- GPU training is fully on `KernelAbstractions.jl`, sharing loss, metric and split-scan math with the CPU path. Oblivious trees (`tree_type=:oblivious`) are supported on GPU.
- New backends besides NVIDIA/CUDA: AMD/ROCm (`AMDGPU.jl`) and Apple Metal (`Metal.jl`). Load the corresponding package, then set `device` on the learner.
- `device` accepts `:cpu`, `:gpu` / `:cuda` (NVIDIA), `:rocm` / `:amd`, and `:metal`. `:gpu` remains an alias for CUDA.

### Other
- `early_stopping_tolerance`: eval metric must improve by more than this value to reset the early-stopping counter (default `0.0`).
- `EvoTrees.predict_leaf_idx(m, data)`: leaf index of each observation in each tree (`Matrix{UInt32}` of size `(nobs, ntrees)`).
- Quantile eval metric now uses the model's `alpha`. `:logistic_mle` and `:gini` eval metrics match their intended definitions.
- `importance` returns zeros when a model has no splits.

## v0.18


## Refactor of GPU training backend
- Computations are now done through `KernelAbstractions.jl` instead of CUDA specific kernels. Objective is to eventually have full support for AMD / ROCm in addition to current NVIDIA / CUDA devices.
- Important performance increase, notably for larger max depth. Training time is now closely increase linearly with depth. 

### Breaking change: improved reproducibility
- Training returns exactly the same fitted  model for a given learner (ex: `EvoTreeRegressor`). 
- Reproducibility is respected for both `cpu` and `gpu`. However, results may differ between `cpu` and `gpu`. Ie: reproducibility is guaranteed only within the same device type.
- The learner / model constructor (ex: `EvoTreeRegressor`) now has a `seed::Int` argument to set the random seed. Legacy `rng` kwarg will now be ignored.
- The internal random generator is now `Random.Xoshiro` (was previously `Random.MersenneTwister` with `rng::Int`).

### Added node weight information in fitted trees 
- The train weight reaching each of the split/leaf nodes is now stored in the fitted trees. This is accessible via `model.trees[i].w` for the i-th tree in the fitted model. This is notably inteded to support SHAP value computations.

```julia
config = EvoTreeRegressor(; max_depth=3)
m = fit(config; x_train, y_train)
m.trees[2].w

7-element Vector{Float32}:
 8000.0
 5000.0
 3000.0
  750.0
 4250.0
 1250.0
 1750.0
```
