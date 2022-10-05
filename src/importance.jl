function importance!(gain::AbstractVector, tree::Union{Tree,TreeGPU})
    @inbounds for n in eachindex(tree.split)
        if @allowscalar(tree.split[n])
            @allowscalar(gain[tree.feat[n]] += tree.gain[n])
        end
    end
end

"""
    importance(model::GBTree)

Sorted normalized feature importance based on loss function gain.
"""
function importance(model::Union{GBTree,GBTreeGPU})
    fnames = model.info[:fnames]
    gain = zeros(length(fnames))

    # Loop importance over all trees and sort results.
    for tree in model.trees
        importance!(gain, tree)
    end

    gain .= gain ./ sum(gain)
    pairs = collect(Dict(zip(string.(fnames), gain)))
    sort!(pairs, by=x -> -x[2])

    return pairs
end
