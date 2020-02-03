module EvoTrees

export init_evotree, grow_evotree!, grow_tree, predict, fit_evotree,
    EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian,
    EvoTreeRModels

using Statistics
using Base.Threads: @threads
using StatsBase: sample, quantile
import StatsBase: predict
using Random: seed!
using StaticArrays
using Distributions
using CategoricalArrays
import MLJBase
# import MLJ

include("models.jl")
include("structs.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("find_split.jl")
include("fit.jl")
include("MLJ.jl")

end # module
