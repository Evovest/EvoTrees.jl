function importance!(gain::AbstractVector, tree::Tree)
    @inbounds for n in eachindex(tree.split)
        if tree.split[n]
            gain[tree.feat[n]] += tree.gain[n]
        end
    end
end

"""
    importance(model::EvoTree; feature_names=model.info[:feature_names])

Sorted normalized feature importance based on loss function gain.
Feature names associated to the model are stored in `model.info[:feature_names]` as a string `Vector` and can be updated at any time. Eg: `model.info[:feature_names] = new_feature_names_vec`.
"""
function importance(model::EvoTree; feature_names=model.info[:feature_names])
    gain = zeros(length(feature_names))

    for tree in model.trees
        importance!(gain, tree)
    end

    gain .= gain ./ sum(gain)
    pairs = collect(Dict(zip(Symbol.(feature_names), gain)))
    sort!(pairs, by=x -> -x[2])

    return pairs
end
