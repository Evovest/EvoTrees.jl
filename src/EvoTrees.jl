module EvoTrees

export init_evotree, grow_evotree!, grow_tree, fit_evotree, predict,
    EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian,
    EvoTreeRModels, importance, Random

using Base.Threads: @threads
using Statistics
using StatsBase: sample, sample!, quantile
using SpecialFunctions: loggamma
using Random
using Distributions
using StaticArrays
using CategoricalArrays
using CUDA
using CUDA: @allowscalar, allowscalar
using BSON
using NetworkLayout
using RecipesBase
import MLJModelInterface
import MLJModelInterface: fit, update, predict, schema
import Base: convert

include("models.jl")
include("structs.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("find_split.jl")
include("fit.jl")

include("gpu/structs_gpu.jl")
include("gpu/loss_gpu.jl")
include("gpu/eval_gpu.jl")
include("gpu/predict_gpu.jl")
include("gpu/find_split_gpu.jl")
include("gpu/fit_gpu.jl")

include("importance.jl")
include("plot.jl")
include("MLJ.jl")

function convert(::Type{GBTree}, m::GBTreeGPU)
    EvoTrees.GBTree([EvoTrees.Tree(Array(tree.feat),
            Array(tree.cond_bin),
            Array(tree.cond_float),
            Array(tree.gain),
            Array(tree.pred),
            Array(tree.split)) for tree in m.trees],
        m.params,
        m.metric,
        m.K,
        m.levels)
end

function save(model::GBTree, path)
    BSON.bson(path, Dict(:model => model))
end

function save(model::GBTreeGPU, path)
    m = convert(GBTree, model)
    save(m, path)
end

function load(path)
    m = BSON.load(path, @__MODULE__)
    return m[:model]
end

end # module
