# NEWS

## v0.18

## Refactor of GPU training backend
- Computations are now alsmost entirely done through `KernelAbstractions.jl`. Objective is to eventually have full support for AMD / ROCm in addition to current NVIDIA / CUDA devices.
- Important performance increase, notably for larger max depth. Training time is now closely increase linearly with depth. 

### Breaking change: improved reproducibility
- Training returns exactly the same fitted  model for a given learner (ex: `EvoTreeRegressor`). 
- Reproducibility is respected for both `cpu` and `gpu`. However, thes result may differ between `cpu` and `gpu`. Ie: reproducibility is guaranteed only within the same device type.
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
