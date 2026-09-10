module EvoTrees

# export fit_evotree
export EvoTreeRegressor,
    EvoTreeCount,
    EvoTreeClassifier,
    EvoTreeMLE,
    EvoTreeGaussian,
    EvoTree,
    treeplot,
    treeplot!

using Base.Threads: @threads, @spawn, nthreads
using Statistics
using StatsBase: sample, sample!, quantile, proportions
using Random
using Random: seed!, AbstractRNG, Xoshiro
import Distributions
using Tables
using CategoricalArrays
using Tables
using BSON

using MLJModelInterface
import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema, feature_importances
import Base: convert
import Base: depwarn

include("groups.jl")
include("learners.jl")
include("loss.jl")
include("split.jl")
include("metrics.jl")
include("structs.jl")
include("predict.jl")
include("init.jl")
include("subsample.jl")
include("fit-utils.jl")
include("fit.jl")
include("callback.jl")

include("MLJ.jl")

include("importance.jl")
include("shap.jl")
using .Shap
include("plot.jl")

"""
    EvoTreeLegacyPreV20{L,K}

The `EvoTree` layout used before v0.20 moved the intercept out of `trees[1]` into `bias`. It exists
only so `load` can read a model saved by an earlier version; nothing else should reach for it.
"""
struct EvoTreeLegacyPreV20{L,K}
    loss_type::Type{L}
    K::UInt8
    trees::Vector{Tree{L,K}}
    info::Dict{Symbol,Any}
end

function save(model::EvoTree{L,K}, path) where {L,K}
    # stamped so a later layout change has something to branch on rather than guessing from shape
    info = copy(model.info)
    info[:save_version] = string(pkgversion(@__MODULE__))
    stamped = EvoTree{L,K}(model.loss_type, model.K, model.bias, model.trees, info)
    BSON.bson(path, Dict(:model => stamped))
end

"""
    load(path)

Read a model written by [`save`](@ref), including one written before v0.20.

BSON rebuilds a struct by field position, so a model from an earlier version puts its `trees` into
the `bias` field and fails to load. Nothing in the file records which layout it is, so it is read
off the document: four stored fields is the old one, whose first tree is a single leaf carrying the
intercept. `save` now records a version, which removes the guesswork for the next change.
"""
function load(path)
    doc = BSON.parse(path)
    entry = get(doc, :model, nothing)
    if _is_pre_v20(entry)
        entry[:type][:name] = Any["EvoTrees", "EvoTreeLegacyPreV20"]
        return _upgrade(BSON.raise_recursive(doc, @__MODULE__)[:model])
    end
    return BSON.load(path, @__MODULE__)[:model]
end

_is_pre_v20(entry) =
    entry isa AbstractDict &&
    haskey(entry, :data) && length(entry[:data]) == 4 &&
    haskey(entry, :type) && entry[:type][:name] == Any["EvoTrees", "EvoTree"]

function _upgrade(m::EvoTreeLegacyPreV20{L,K}) where {L,K}
    isempty(m.trees) && error("This model was saved before v0.20 and carries no trees, so its " *
                              "intercept cannot be recovered.")
    # the first tree was the intercept: a single leaf, no splits
    bias = vec(copy(m.trees[1].pred))
    info = copy(m.info)
    info[:save_version] = get(info, :save_version, "pre-0.20")
    return EvoTree{L,K}(L, m.K, bias, m.trees[2:end], info)
end

end # module

