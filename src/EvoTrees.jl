module EvoTrees

export grow_tree!, grow_gbtree, grow_gbtree!, Tree, Node, Params, predict, EvoTreeRegressor, EvoTreeRegressorR

using DataFrames
using Statistics
using CSV
using Base.Threads: @threads
using StatsBase: sample, quantile
using Random: seed!
using StaticArrays
using CategoricalArrays
using Flux: onehot
import MLJBase
# import MLJ

include("MLJ_struct.jl")
include("struct.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("find_split.jl")
include("trees.jl")
include("MLJ_fit.jl")

end # module
