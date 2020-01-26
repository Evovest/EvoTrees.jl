module EvoTrees

export init_evotree, grow_evotree!, grow_tree, predict, fit_evotree,
    EvoTreeRegressor, EvoTreeCount, EvoTreeClassifier, EvoTreeGaussian,
    EvoTreeRegressorR

using DataFrames
using Statistics
using CSV
using Base.Threads: @threads
using StatsBase: sample, quantile
using Random: seed!
using StaticArrays
using Distributions
using CategoricalArrays
using Flux: onehot
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
