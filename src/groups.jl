"""
    GroupIndex

Row to group (query) mapping for ranking. Groups are given as a per-row id column rather
than boundary offsets, so a group's rows need not be contiguous.

`group` is the normalised id of each row in `1:ngroups`, `rows` holds row indices ordered by
group, and `ptr` bounds them, so group `g` owns `rows[ptr[g]:ptr[g+1]-1]`.
"""
struct GroupIndex
    group::Vector{UInt32}
    rows::Vector{UInt32}
    ptr::Vector{UInt32}
end

ngroups(gi::GroupIndex) = length(gi.ptr) - 1

@inline group_rows(gi::GroupIndex, g) = view(gi.rows, gi.ptr[g]:(gi.ptr[g+1]-one(UInt32)))

Base.length(gi::GroupIndex) = length(gi.group)

"""
    build_group_index(raw::AbstractVector, nobs=nothing, argname="group")

Build a [`GroupIndex`](@ref) from a per-row group id column. Ids may be of any type
supporting `sort` and `isequal`, and need not be contiguous, sorted, or numeric.

`nobs` is checked when given: a short group column would otherwise silently train or score
on a subset, and a long one would index past the end of the predictions.
"""
function build_group_index(raw::AbstractVector, nobs::Union{Nothing,Integer}=nothing, argname::AbstractString="group")
    n = length(raw)
    n == 0 && error("`$(argname)` must be non-empty.")
    if !isnothing(nobs) && n != nobs
        error(
            "`$(argname)` has length $(n) but there are $(nobs) observations. " *
            "Each row needs exactly one group id."
        )
    end
    levels = sort(unique(raw))
    lookup = Dict(lv => UInt32(i) for (i, lv) in enumerate(levels))
    group = Vector{UInt32}(undef, n)
    @inbounds for i in 1:n
        group[i] = lookup[raw[i]]
    end
    return _index_from_ids(group, UInt32(length(levels)))
end

function _index_from_ids(group::Vector{UInt32}, ng::UInt32)
    counts = zeros(UInt32, ng)
    @inbounds for g in group
        counts[g] += one(UInt32)
    end
    ptr = Vector{UInt32}(undef, ng + 1)
    ptr[1] = one(UInt32)
    @inbounds for g in 1:ng
        ptr[g+1] = ptr[g] + counts[g]
    end
    cursor = copy(ptr)
    rows = Vector{UInt32}(undef, length(group))
    @inbounds for i in eachindex(group)
        g = group[i]
        rows[cursor[g]] = UInt32(i)
        cursor[g] += one(UInt32)
    end
    return GroupIndex(group, rows, ptr)
end
