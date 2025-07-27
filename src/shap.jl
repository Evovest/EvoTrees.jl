function treeshapv1(m::EvoTrees.EvoTree, data; feature_names=m.info[:feature_names])
    Tables.istable(data) ? data = Tables.columntable(data) : nothing
    x_bin = binarize(data; feature_names=m.info[:feature_names], edges=m.info[:edges])
    nobs, nfeats = size(x_bin)
    @info "info" nobs, nfeats
    @info "size(x_bin)" size(x_bin)
    @info "extrema(x_bin))" Int.(extrema(x_bin))

    shap = rand(Float32, nobs, nfeats)
    @info "size(x_bin)" size(x_bin)
    for n in 1:nobs
        @info "n" n
        @info "x_bin[n, :]" x_bin[n, :]
        view(shap, n, :) .= treeshapv1(m.trees, x_bin[n, :])
    end
    # shap = treeshapv1(m.trees, x_bin[1, :])
    return shap
end

"""
FastTreeSHAPv1 SHAP values for a EvoTrees.EvoTree model

## References
* Jilei Yang, [*Fast TreeSHAP: Accelerating SHAP Value Computation for Trees*](https://arxiv.org/abs/2109.09847), 2022, pp. 16-17
"""
function treeshapv1(trees::Vector{<:Tree}, x::Vector{UInt8})
    shap = treeshapv1(trees[1], x)
    for i in 2:length(trees)
        shap .+= treeshapv1(trees[i], x)
    end
    # shap = treeshapv1(trees[i], x)
    return shap
end

"""
FastTreeSHAPv1 SHAP values for a single EvoTrees.Tree

## References
* Jilei Yang, [*Fast TreeSHAP: Accelerating SHAP Value Computation for Trees*](https://arxiv.org/abs/2109.09847), 2022, pp. 16-17
"""
function treeshapv1(tree::EvoTrees.Tree, x::AbstractVector)
    φ = zeros(Float32, length(x))
    recurse!(tree, firstindex(tree.feat), PathValue[], Float32[], 1., PathValue(0, 1., true), φ, x)
    return φ
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
    z::Float32
    o::Bool
end

function nodevalue(tree::EvoTrees.Tree{L,K}, j::Integer) where {L<:EvoTrees.GradientRegression,K}
    tree.pred[1, j]
end

function nodevalue(tree::EvoTrees.Tree{L,K}, j::Integer) where {L<:Union{EvoTrees.MLogLoss},K}
    sum(tree.pred[:, j]) # logit sum
end

function recurse!(tree::EvoTrees.Tree, j::Integer, m::AbstractVector{PathValue}, w::AbstractVector, q::Real, p::PathValue, φ::AbstractVector, x::AbstractVector)
    m, w, q = extend(m, w, q, p)

    if !tree.split[j] # leaf node
        if length(m) > length(w)
            s₀ = -sum(unwind(m, w, q)[2])
        end

        for i = 2:lastindex(m)
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
        aⱼ, bⱼ = (2j, 2j + 1) # j's (left, right) child
        h, c = x[dⱼ] ≤ tree.cond_bin[j] ? (aⱼ, bⱼ) : (bⱼ, aⱼ) # (hot, cold) child
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
        for i = lw:-1:1
            w[i] *= (l - (i - 1)) / (l + 1)
        end
    else
        push!(w, T(lw == 0))
        for i = lw:-1:1
            w[i+1] += w[i] * (i / (l + 1))
            w[i] *= p.z * ((l - (i - 1)) / (l + 1))
        end
    end
    m, w, q
end

function unwind(m::AbstractVector{PathValue}, w::AbstractVector, q::Real)
    l, lw = length(m) - 1, length(w) - 1
    m, w = copy(view(m, 1:l)), copy(w)

    for j = lw+1:-1:1
        w[j] *= (l + 1) / (l - (j - 1))
    end
    m, w, q
end

function unwind(m::AbstractVector{PathValue}, w::AbstractVector, q::Real, i::Integer)
    l, lw = length(m) - 1, length(w) - 1
    mᵢ = m[i]
    m = copy(view(m, 1:l))

    if !mᵢ.o
        w = copy(w)
        for j = lw+1:-1:1
            w[j] *= (l + 1) / (l - (j - 1))
        end
        q /= mᵢ.z
    else
        n = w[lw+1]
        w = copy(view(w, 1:lw))
        for j = lw:-1:1
            t = w[j]
            w[j] = n * ((l + 1) / (j + 1))
            n = t - w[j] * mᵢ.z * ((l - (j - 1)) / (l + 1))
        end
    end
    for j = i:l-1
        m[j] = m[j+1]
    end
    m, w, q
end

