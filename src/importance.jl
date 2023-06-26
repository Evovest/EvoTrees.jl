function importance!(gain::AbstractVector, tree::Tree)
    @inbounds for n in eachindex(tree.split)
        if tree.split[n]
            gain[tree.feat[n]] += tree.gain[n]
        end
    end
end

"""
    importance(model::EvoTree; fnames=model.info[:fnames])

Sorted normalized feature importance based on loss function gain.
Feature names associated to the model are stored in `model.info[:fnames]` as a string `Vector` and can be updated at any time. Eg: `model.info[:fnames] = new_fnames_vec`.
"""
function importance(model::EvoTree; fnames=model.info[:fnames])
    gain = zeros(length(fnames))

    for tree in model.trees
        importance!(gain, tree)
    end

    gain .= gain ./ sum(gain)
    pairs = collect(Dict(zip(string.(fnames), gain)))
    sort!(pairs, by=x -> -x[2])

    return pairs
end
