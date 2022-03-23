# importance from single tree
# function importance!(gain::AbstractVector, tree::Tree)
#     @inbounds for node in tree.nodes
#         if node.split
#             gain[node.feat] += node.gain
#         end
#     end
# end

function importance!(gain::AbstractVector, tree::Union{Tree,TreeGPU})
    @inbounds for n in eachindex(tree.split)
        if @allowscalar(tree.split[n])
            @allowscalar(gain[tree.feat[n]] += tree.gain[n])
        end
    end
end

"""
    importance(model::GBTree, vars::AbstractVector)

Sorted normalized feature importance based on loss function gain.
"""
function importance(model::Union{GBTree,GBTreeGPU}, vars::AbstractVector)
    gain = zeros(length(vars))

    # Loop importance over all trees and sort results.
    for tree in model.trees
        importance!(gain, tree)
    end

    gain .= gain ./ sum(gain)
    pairs = collect(Dict(zip(string.(vars), gain)))
    sort!(pairs, by = x -> -x[2])

    return pairs
end

