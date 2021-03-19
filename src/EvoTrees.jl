module EvoTrees

export init_evotree, grow_evotree!, grow_tree, fit_evotree, predict,
    # fit_evotree_gpu, predict_gpu,
    EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian,
    EvoTreeRModels, importance, Random

using Base.Threads: @threads
using Statistics
using StatsBase: sample, quantile
using Random
using Distributions
using StaticArrays
using CategoricalArrays
using CUDA
using BSON: @save, @load
using NetworkLayout
using RecipesBase
import MLJModelInterface
import MLJModelInterface: fit, update, predict, schema

include("models.jl")
include("structs.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("find_split.jl")
include("fit.jl")
include("importance.jl")
include("plot.jl")
include("MLJ.jl")

include("gpu/structs_gpu.jl")
include("gpu/loss_gpu.jl")
include("gpu/eval_gpu.jl")
include("gpu/predict_gpu.jl")
include("gpu/find_split_gpu.jl")
include("gpu/fit_gpu.jl")

function save(model::GBTree, path)
    @save path model
end

function load(path)
    @load path model
    return model
end

end # module
