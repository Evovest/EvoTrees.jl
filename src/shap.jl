"""
FastTreeSHAPv1 SHAP values for a EvoTrees.EvoTree model

## References
* Jilei Yang, [*Fast TreeSHAP: Accelerating SHAP Value Computation for Trees*](https://arxiv.org/abs/2109.09847), 2022, pp. 16-17
"""
function treeshapv1(model::EvoTrees.EvoTree, x::AbstractVector; fnames=model.info[:fnames])
    chan = Channel{typeof(x)}(length(model.trees))
    @inbounds Threads.@threads for tree in model.trees
        put!(chan, treeshapv1!(tree, zero(x), x))
    end
    ϕ = take!(chan)
    while !isempty(chan)
        ϕ += take!(chan)
    end

    pairs = string.(fnames) .=> ϕ
    sort!(pairs, by=x->abs(x[2]), rev=true)
    pairs
end

"""
FastTreeSHAPv1 SHAP values for a single EvoTrees.Tree

## References
* Jilei Yang, [*Fast TreeSHAP: Accelerating SHAP Value Computation for Trees*](https://arxiv.org/abs/2109.09847), 2022, pp. 16-17
"""
function treeshapv1!(tree::EvoTrees.Tree, φ::AbstractVector, x::AbstractVector)
    recurse!(tree, firstindex(tree.feat), PathValue[], Float64[], 1., PathValue(0, 1., true), φ, x)
    φ
end

"""
Element type used to make *m*, the path vector of unique features we have split on so far.

## Fields
* `d`: feature index of the jth internal node
* `z`: covering ratio for the jth internal node
* `o`: the threshold condition for the jth internal node
"""
struct PathValue
    d::Int
    z::Float64
    o::Bool
end

function nodevalue(tree::EvoTrees.Tree{L, K}, j::Integer) where {L<:EvoTrees.GradientRegression, K}
    tree.pred[1, j]
end

function nodevalue(tree::EvoTrees.Tree{L, K}, j::Integer) where {L<:Union{EvoTrees.LogLoss,EvoTrees.MLogLoss}, K}
    sum(tree.pred[:, j]) # logit sum
end

function recurse!(tree::EvoTrees.Tree, j::Integer, m::AbstractVector{PathValue}, w::AbstractVector, q::Real, p::PathValue, φ::AbstractVector, x::AbstractVector)
    m, w, q = extend(m, w, q, p)

    if !tree.split[j] # leaf node
        if length(m) > length(w)
            s₀ = -sum(unwind(m, w, q)[2])
        end

        for i=2:lastindex(m)
            vⱼ = nodevalue(tree, j)
            if !m[i].o
                update = s₀ * q * vⱼ
            else
                s = sum(unwind(m, w, q, i)[2])
                update = s * q * (1 - m[i].z) * vⱼ
            end
            φ[m[i].d] += update
        end
    else
        dⱼ = tree.feat[j]
        aⱼ, bⱼ = (2j, 2j+1) # j's (left, right) child
        h, c = x[dⱼ] ≤ tree.cond_float[j] ? (aⱼ, bⱼ) : (bⱼ, aⱼ) # (hot, cold) child
        iz, io = 1, true
        k = findfirst(x -> x.d == dⱼ, m)

        if !isnothing(k)  # Undo previous updates if feature was already used
            iz, io = m[k].z, m[k].o
            m, w, q = unwind(m, w, q, k)
        end
        recurse!(tree, h, m, w, q, PathValue(dⱼ, iz * (tree.gain[h] / tree.gain[j]), io), φ, x)
        recurse!(tree, c, m, w, q, PathValue(dⱼ, iz * (tree.gain[c] / tree.gain[j]), false), φ, x)
    end
end

function extend(m::AbstractVector{PathValue}, w::AbstractVector{T}, q::Real, p::PathValue) where {T<:Real}
    l, lw = length(m), length(w)
    m, w = copy(m), copy(w)
    push!(m, p)

    if !p.o
        q *= p.z
        for i=lw:-1:1
            w[i] *= (l - (i - 1)) / (l + 1)
        end
    else
        push!(w, T(lw == 0))
        for i=lw:-1:1
            w[i + 1] += w[i] * (i / (l + 1))
            w[i] *= p.z * ((l - (i - 1)) / (l + 1))
        end
    end
    m, w, q
end

function unwind(m::AbstractVector{PathValue}, w::AbstractVector, q::Real)
    l, lw = length(m)-1, length(w)-1
    m, w = copy(view(m, 1:l)), copy(w)

    for j=lw+1:-1:1
        w[j] *= (l + 1) / (l - (j - 1))
    end
    m, w, q
end

function unwind(m::AbstractVector{PathValue}, w::AbstractVector, q::Real, i::Integer)
    l, lw = length(m)-1, length(w)-1
    mᵢ = m[i]
    m = copy(view(m, 1:l))

    if !mᵢ.o
        w = copy(w)
        for j=lw+1:-1:1
            w[j] *= (l + 1) / (l - (j - 1))
        end
        q /= mᵢ.z
    else
        n = w[lw+1]
        w = copy(view(w, 1:lw))
        for j=lw:-1:1
            t = w[j]
            w[j] = n * ((l + 1) / (j + 1))
            n = t - w[j] * mᵢ.z * ((l - (j - 1)) / (l + 1))
        end
    end
    for j=i:l-1
        m[j] = m[j + 1]
    end
    m, w, q
end

