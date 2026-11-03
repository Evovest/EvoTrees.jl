@inline _term_gh(g, h, w, lambda, L2, ϵ) = g^2 / max(ϵ, h + lambda * w + L2) / 2
@inline _term_shift(g, w, μp, d) = abs(g / w - μp) * w / d
@inline _term_cred(::Type{L}, m1, m2, w, L2, ϵ) where {L<:Cred} =
    _cred_Z(L, m1, m2, w, ϵ) * abs(m1) / (1 + L2 / w)

"""
    _acc_left!(∑L, h∇, j, bin, n, is_numeric)
    _acc_left!(sums, col, h∇, j, bin, n, NK, is_numeric)

Add histogram bin `bin` of feature `j` at node `n` into the left-side stats.
Numeric features accumulate; categorical features replace with that bin only.

The 1D method writes `∑L`. The 2D method writes column `col` of `sums`
(`NK == 2K + 1`).
"""
Base.@propagate_inbounds function _acc_left!(∑L, h∇, j, bin, n, is_numeric)
    if is_numeric
        for k in eachindex(∑L)
            ∑L[k] += h∇[k, bin, j, n]
        end
    else
        for k in eachindex(∑L)
            ∑L[k] = h∇[k, bin, j, n]
        end
    end
    return nothing
end

Base.@propagate_inbounds function _acc_left!(sums, col::Integer, h∇, j, bin, n, NK::Integer, is_numeric)
    for k in 1:NK
        hk = h∇[k, bin, j, n]
        sums[k, col] = is_numeric ? sums[k, col] + hk : oftype(sums[k, col], hk)
    end
    return nothing
end

"""
    _accumulate_hist_k1(h∇, f, b, node, is_numeric, acc1, acc2, accw)

Register accumulate for `K == 1` (`g`, `h`, `w`). Same numeric/categorical rule
as `_acc_left!`.
"""
Base.@propagate_inbounds function _accumulate_hist_k1(h∇, f, b, node, is_numeric, acc1, acc2, accw)
    if is_numeric
        return (acc1 + h∇[1, b, f, node], acc2 + h∇[2, b, f, node], accw + h∇[3, b, f, node])
    else
        return (h∇[1, b, f, node], h∇[2, b, f, node], h∇[3, b, f, node])
    end
end

"""
    node_gain(::Type{L}, ∑, lambda, L2, ϵ)
    node_gain(::Type{L}, nodes_sum, node, K, lambda, L2, ϵ)

Parent-node gain from stats `∑` (1D) or column `node` of `nodes_sum` (2D).
`ϵ` is supplied by the caller (`eps(T)` on CPU, `1e-8` on GPU).
"""
@inline function node_gain(::Type{L}, ∑, lambda, L2, ϵ::T) where {T,L<:Union{GradientRegression,MLE2P,MLogLoss}}
    K = (length(∑) - 1) ÷ 2
    w = ∑[2K+1]
    gain = zero(T)
    @inbounds for k in 1:K
        gain += _term_gh(∑[k], ∑[K+k], w, lambda, L2, ϵ)
    end
    return gain
end

@inline node_gain(::Type{L}, ∑, lambda, L2, ϵ::T) where {T,L<:Union{MAE,Quantile}} = zero(T)

@inline function node_gain(::Type{L}, ∑, lambda, L2, ϵ::T) where {T,L<:Cred}
    K = (length(∑) - 1) ÷ 2
    w = ∑[2K+1]
    gain = zero(T)
    @inbounds for k in 1:K
        gain += _term_cred(L, ∑[k], ∑[K+k], w, L2, ϵ)
    end
    return gain
end

Base.@propagate_inbounds function node_gain(::Type{L}, nodes_sum, node::Integer, K::Int, lambda, L2, ϵ::T) where {T,L<:Union{GradientRegression,MLE2P,MLogLoss}}
    w = nodes_sum[2K+1, node]
    gain = zero(T)
    for k in 1:K
        gain += _term_gh(nodes_sum[k, node], nodes_sum[K+k, node], w, lambda, L2, ϵ)
    end
    return gain
end

Base.@propagate_inbounds node_gain(::Type{L}, nodes_sum, node::Integer, K::Int, lambda, L2, ϵ::T) where {T,L<:Union{MAE,Quantile}} = zero(T)

Base.@propagate_inbounds function node_gain(::Type{L}, nodes_sum, node::Integer, K::Int, lambda, L2, ϵ::T) where {T,L<:Cred}
    w = nodes_sum[2K+1, node]
    gain = zero(T)
    for k in 1:K
        gain += _term_cred(L, nodes_sum[k, node], nodes_sum[K+k, node], w, L2, ϵ)
    end
    return gain
end

"""
    split_gain(::Type{L}, ∑, ∑L, w_l, w_r, lambda, L2, ϵ)
    split_gain(::Type{L}, nodes_sum, node, sums, col, K, w_l, w_r, lambda, L2, ϵ)

Left + right child gain. Right stats are `∑ - ∑L` (1D) or
`nodes_sum[:, node] - sums[:, col]` (2D). Net split gain is this value minus
`node_gain` of the parent.
"""
@inline function split_gain(::Type{L}, ∑, ∑L, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Union{GradientRegression,MLE2P,MLogLoss}}
    K = (length(∑) - 1) ÷ 2
    gain = zero(T)
    @inbounds for k in 1:K
        g_l, h_l = ∑L[k], ∑L[K+k]
        gain += _term_gh(g_l, h_l, w_l, lambda, L2, ϵ) +
                _term_gh(∑[k] - g_l, ∑[K+k] - h_l, w_r, lambda, L2, ϵ)
    end
    return gain
end

@inline function split_gain(::Type{L}, ∑, ∑L, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Union{MAE,Quantile}}
    K = (length(∑) - 1) ÷ 2
    w_p = ∑[2K+1]
    d_l = max(ϵ, 1 + lambda + L2 / w_l)
    d_r = max(ϵ, 1 + lambda + L2 / w_r)
    gain = zero(T)
    @inbounds for k in 1:K
        μp = ∑[k] / w_p
        gain += _term_shift(∑L[k], w_l, μp, d_l) + _term_shift(∑[k] - ∑L[k], w_r, μp, d_r)
    end
    return gain
end

@inline function split_gain(::Type{L}, ∑, ∑L, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Cred}
    K = (length(∑) - 1) ÷ 2
    gain = zero(T)
    @inbounds for k in 1:K
        m1l, m2l = ∑L[k], ∑L[K+k]
        gain += _term_cred(L, m1l, m2l, w_l, L2, ϵ) +
                _term_cred(L, ∑[k] - m1l, ∑[K+k] - m2l, w_r, L2, ϵ)
    end
    return gain
end

Base.@propagate_inbounds function split_gain(::Type{L}, nodes_sum, node::Integer, sums, col::Integer, K::Int, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Union{GradientRegression,MLE2P,MLogLoss}}
    gain = zero(T)
    for k in 1:K
        g_l, h_l = sums[k, col], sums[K+k, col]
        gain += _term_gh(g_l, h_l, w_l, lambda, L2, ϵ) +
                _term_gh(nodes_sum[k, node] - g_l, nodes_sum[K+k, node] - h_l, w_r, lambda, L2, ϵ)
    end
    return gain
end

Base.@propagate_inbounds function split_gain(::Type{L}, nodes_sum, node::Integer, sums, col::Integer, K::Int, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Union{MAE,Quantile}}
    w_p = nodes_sum[2K+1, node]
    d_l = max(ϵ, 1 + lambda + L2 / w_l)
    d_r = max(ϵ, 1 + lambda + L2 / w_r)
    gain = zero(T)
    for k in 1:K
        μp = nodes_sum[k, node] / w_p
        g_l = sums[k, col]
        gain += _term_shift(g_l, w_l, μp, d_l) + _term_shift(nodes_sum[k, node] - g_l, w_r, μp, d_r)
    end
    return gain
end

Base.@propagate_inbounds function split_gain(::Type{L}, nodes_sum, node::Integer, sums, col::Integer, K::Int, w_l::T, w_r::T, lambda, L2, ϵ::T) where {T,L<:Cred}
    gain = zero(T)
    for k in 1:K
        m1l, m2l = sums[k, col], sums[K+k, col]
        gain += _term_cred(L, m1l, m2l, w_l, L2, ϵ) +
                _term_cred(L, nodes_sum[k, node] - m1l, nodes_sum[K+k, node] - m2l, w_r, L2, ϵ)
    end
    return gain
end

"""
    _init_split_scan(L, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints, lambda, L2, K, ε)

Return `(f, is_numeric, constraint, w_p, gain_p, b_max)` for one `(node, feature)`.
Numeric features scan bins `1:nbins-1`; categorical features scan `1:nbins`.
"""
Base.@propagate_inbounds function _init_split_scan(
    ::Type{L}, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints,
    lambda::T, L2::T, K::Int, ε::T,
) where {T,L}
    f = js[f_idx]
    is_numeric = feattypes[f]
    constraint = monotone_constraints[f]
    w_p = nodes_sum[2*K+1, node]
    gain_p = node_gain(L, nodes_sum, node, K, lambda, L2, ε)
    nbins = size(h∇, 2)
    b_max = is_numeric ? (nbins - 1) : nbins
    return f, is_numeric, constraint, w_p, gain_p, b_max
end

"""
    _clear_split_sums!(sums_temp, temp_idx, K)

Zero column `temp_idx` of `sums_temp` before a `K > 1` feature scan.
"""
Base.@propagate_inbounds function _clear_split_sums!(sums_temp, temp_idx, K)
    if K > 1
        for kk in 1:(2*K+1)
            sums_temp[kk, temp_idx] = zero(eltype(sums_temp))
        end
    end
    return nothing
end
