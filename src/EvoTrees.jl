module EvoTrees

export init_evotree, grow_evotree!, fit_evotree, importance
export EvoTreeRegressor,
    EvoTreeCount,
    EvoTreeClassifier,
    EvoTreeMLE,
    EvoTreeGaussian,
    EvoTree,
    EvoTreeGPU,
    Random

using Base.Threads: @threads
using Statistics
using StatsBase: sample, sample!, quantile
using Random
using Distributions
using CategoricalArrays
using LoopVectorization
using Tables
using CUDA
using CUDA: @allowscalar, allowscalar
using BSON

using NetworkLayout
using RecipesBase

using MLJModelInterface
import MLJModelInterface as MMI
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

include("callback.jl")
include("importance.jl")
include("plot.jl")
include("MLJ.jl")

function convert(::Type{EvoTree}, m::EvoTreeGPU{L,K,T}) where {L,K,T}
    EvoTrees.EvoTree{L,K,T}(
        [
            EvoTrees.Tree{L,K,T}(
                Array(tree.feat),
                Array(tree.cond_bin),
                Array(tree.cond_float),
                Array(tree.gain),
                Array(tree.pred),
                Array(tree.split),
            ) for tree in m.trees
        ],
        m.info,
    )
end

function save(model::EvoTree, path)
    BSON.bson(path, Dict(:model => model))
end

function save(model::EvoTreeGPU, path)
    m = convert(EvoTree, model)
    save(m, path)
end

function load(path)
    m = BSON.load(path, @__MODULE__)
    return m[:model]
end

end # module
