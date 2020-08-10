module EvoTrees

export init_evotree, grow_evotree!, grow_tree, fit_evotree, predict,
    fit_evotree_gpu, predict_gpu,
    EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian,
    EvoTreeRModels, importance

using Statistics
using Base.Threads: @threads
using StatsBase: sample, quantile
using Random
using StaticArrays
using Distributions
using CategoricalArrays
using CUDA
import MLJModelInterface
import MLJModelInterface: fit, update
import MLJModelInterface: predict

include("models.jl")
include("structs.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("find_split.jl")
include("fit.jl")
include("importance.jl")
include("MLJ.jl")

include("structs_gpu.jl")
include("loss_gpu.jl")
include("eval_gpu.jl")
include("predict_gpu.jl")
include("find_split_gpu.jl")
include("fit_gpu.jl")

end # module
