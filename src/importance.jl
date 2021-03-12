# importance from single tree
function importance!(gain::AbstractVector, tree::Tree)
    @inbounds for node in tree.nodes
        if node.split
            gain[node.feat] += node.gain
        end
    end
end

# loop importance over all trees and sort results
function importance(model::GBTree, vars::AbstractVector)
    gain = zeros(length(vars))
    for tree in model.trees
        importance!(gain, tree)
    end
    gain .= gain ./ sum(gain)
    pairs = collect(Dict(zip(string.(vars),gain)))
    sort!(pairs, by = x -> -x[2])
    return pairs
end
