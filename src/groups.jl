"""
    GroupIndex

Row to group mapping for ranking tasks.

A group (a query, in learning-to-rank terms) is a set of rows that are ranked against each
other. Groups are supplied as a per-row id column rather than as boundary offsets, so the
rows of a group need not be contiguous in the input. The index built here recovers the rows
of each group in O(1), which is what both group-aware sampling and per-group metrics need.

- `group` holds the normalised group id of each row, in `1:ngroups`.
- `rows` holds row indices ordered by group.
- `ptr` holds the boundaries into `rows`, so group `g` owns `rows[ptr[g]:ptr[g+1]-1]`.
"""
struct GroupIndex
    group::Vector{UInt32}
    rows::Vector{UInt32}
    ptr::Vector{UInt32}
end

"""
    ngroups(gi::GroupIndex)

Number of distinct groups.
"""
ngroups(gi::GroupIndex) = length(gi.ptr) - 1

"""
    group_rows(gi::GroupIndex, g)

Row indices belonging to group `g`, as a view.
"""
@inline group_rows(gi::GroupIndex, g) = view(gi.rows, gi.ptr[g]:(gi.ptr[g+1]-one(UInt32)))

Base.length(gi::GroupIndex) = length(gi.group)

"""
    build_group_index(raw::AbstractVector, nobs=nothing, argname="group")

Build a [`GroupIndex`](@ref) from a per-row group id column. Ids may be of any type
supporting `sort` and `isequal`, and need not be contiguous, sorted, or numeric.

`nobs` is the number of observations the ids must cover. It is checked, because a group
column of the wrong length would otherwise be accepted: a short one silently scores or
trains on a subset, and a long one indexes past the end of the predictions.
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

# Counting sort of row indices by group id, giving `rows` and `ptr`.
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
