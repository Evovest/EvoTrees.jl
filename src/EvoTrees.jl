module EvoTrees

# export fit_evotree
export EvoTreeRegressor,
    EvoTreeCount,
    EvoTreeClassifier,
    EvoTreeMLE,
    EvoTreeGaussian,
    EvoTree

using Base.Threads: @threads, @spawn, nthreads
using Statistics
using StatsBase: sample, sample!, quantile, proportions
using Random
using Random: seed!, AbstractRNG
import Distributions
using Tables
using CategoricalArrays
using Tables
using BSON

using NetworkLayout
using RecipesBase

using MLJModelInterface
import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema, feature_importances
import Base: convert
import Base: depwarn

include("learners.jl")
include("loss.jl")
include("metrics.jl")
include("structs.jl")
include("predict.jl")
include("init.jl")
include("subsample.jl")
include("fit-utils.jl")
include("fit.jl")

# if !isdefined(Base, :get_extension)
#     include("../ext/EvoTreesCUDAExt/EvoTreesCUDAExt.jl")
# end

include("callback.jl")
include("importance.jl")
include("plot.jl")
include("MLJ.jl")

function save(model::EvoTree, path)
    BSON.bson(path, Dict(:model => model))
end

function load(path)
    m = BSON.load(path, @__MODULE__)
    return m[:model]
end

end # module
