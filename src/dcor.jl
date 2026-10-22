# Distance covariance of two vectors and its gradient in O(n log n), after Chaudhuri & Hu (2019).
# See docs/src/dcov-design.md.

struct Fenwick
    t::Vector{Float64}
end
Fenwick(n::Int) = Fenwick(zeros(n))
function fadd!(f::Fenwick, i::Int, v::Float64)
    n = length(f.t)
    @inbounds while i <= n
        f.t[i] += v
        i += i & (-i)
    end
end
function fsum(f::Fenwick, i::Int)
    s = 0.0
    @inbounds while i > 0
        s += f.t[i]
        i -= i & (-i)
    end
    s
end

# Sum in index order: the last bits of `sum` can differ across CPUs.
function _ordered_sum(x)
    s = 0.0
    @inbounds for v in x
        s += Float64(v)
    end
    return s
end

# `(mean, std)` of a vector, summed in index order from its first value.
function _ordered_mean_std(x)
    n = length(x)
    x0 = Float64(first(x))
    s = 0.0
    @inbounds for xi in x
        s += Float64(xi) - x0
    end
    dm = s / n
    v = 0.0
    @inbounds for xi in x
        d = (Float64(xi) - x0) - dm
        v += d * d
    end
    return x0 + dm, sqrt(v / (n - 1))
end

# Vector of length n: sum_j |x_i - x_j| for each i.
function _rowsums(x::AbstractVector)
    n = length(x)
    ord = sortperm(x)
    x0 = Float64(x[ord[1]])   # shifted to the minimum for precision
    xs = [Float64(x[i]) - x0 for i in ord]
    cs = cumsum(xs)
    out = zeros(n)
    @inbounds for (k, i) in enumerate(ord)
        out[i] = (k * xs[k] - cs[k]) + ((cs[n] - cs[k]) - (n - k) * xs[k])
    end
    out
end

# Vectors `L`, `R` of length n: L_i = sum over x_j < x_i of |y_i - y_j|, R_i the same over x_j > x_i.
function _signed_ydist(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    y0 = Float64(minimum(y))
    L = zeros(n)
    R = zeros(n)
    for (out, order) in ((L, ordx), (R, reverse(ordx)))
        cnt = Fenwick(n)
        sy = Fenwick(n)
        k = 1
        while k <= n
            # tied x go to neither side
            j = k
            @inbounds while j < n && x[order[j+1]] == x[order[k]]
                j += 1
            end
            @inbounds for t in k:j
                i = order[t]
                r = ry[i]
                yi = Float64(y[i]) - y0
                c_lo = fsum(cnt, r); s_lo = fsum(sy, r)
                c_all = fsum(cnt, n); s_all = fsum(sy, n)
                out[i] = (c_lo * yi - s_lo) + ((s_all - s_lo) - (c_all - c_lo) * yi)
            end
            @inbounds for t in k:j
                i = order[t]
                fadd!(cnt, ry[i], 1.0)
                fadd!(sy, ry[i], Float64(y[i]) - y0)
            end
            k = j + 1
        end
    end
    L, R
end

# Scalar: sum_ij |x_i - x_j| |y_i - y_j|.
function _cross_term(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    cnt = Fenwick(n); sy = Fenwick(n); sx = Fenwick(n); sxy = Fenwick(n)
    x0 = Float64(minimum(x)); y0 = Float64(minimum(y))
    tot = 0.0
    for i in ordx
        r = ry[i]; xi = Float64(x[i]) - x0; yi = Float64(y[i]) - y0
        c_lo = fsum(cnt, r); sy_lo = fsum(sy, r); sx_lo = fsum(sx, r); sxy_lo = fsum(sxy, r)
        c_all = fsum(cnt, n); sy_all = fsum(sy, n); sx_all = fsum(sx, n); sxy_all = fsum(sxy, n)
        c_hi = c_all - c_lo; sy_hi = sy_all - sy_lo; sx_hi = sx_all - sx_lo; sxy_hi = sxy_all - sxy_lo
        tot += (c_lo * xi * yi - yi * sx_lo - xi * sy_lo + sxy_lo) +
               (xi * sy_hi - sxy_hi - c_hi * xi * yi + yi * sx_hi)
        fadd!(cnt, r, 1.0); fadd!(sy, r, yi); fadd!(sx, r, xi); fadd!(sxy, r, xi * yi)
    end
    2 * tot
end

"""
    dcov2(x::AbstractVector, y::AbstractVector)

Unbiased squared distance covariance of `x` and `y`, both of length `n >= 4`. Returns a scalar.
"""
function dcov2(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dcov2 needs two vectors of the same length, got $n and $(length(y))."))
    n >= 4 || error("The unbiased distance covariance is undefined below 4 observations, got $n.")
    ai = _rowsums(x); bi = _rowsums(y)
    aa = sum(ai); bb = sum(bi)
    S1 = _cross_term(x, y)
    S2 = sum(ai .* bi)
    (S1 - 2 * S2 / (n - 2) + aa * bb / ((n - 1) * (n - 2))) / (n * (n - 3))
end

"""
    dcor2(x::AbstractVector, y::AbstractVector)

Squared distance correlation of `x` and `y`, both of length `n >= 4`:
`dcov2(x, y) / sqrt(dcov2(x, x) * dcov2(y, y))`, or 0 if either variance is not positive.
"""
function dcor2(x::AbstractVector, y::AbstractVector)
    vx = dcov2(x, x)
    vy = dcov2(y, y)
    (vx > 0 && vy > 0) || return 0.0
    return dcov2(x, y) / sqrt(vx * vy)
end

"""
    dcov2_grad(x::AbstractVector, y::AbstractVector)

Gradient of `dcov2(x, y)` with respect to `x`, a vector of length `n`.
"""
function dcov2_grad(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dcov2_grad needs two vectors of the same length, got $n and $(length(y))."))
    n >= 4 || error("The unbiased distance covariance is undefined below 4 observations, got $n.")
    bi = _rowsums(y); bb = sum(bi)
    L, R = _signed_ydist(x, y)
    ordx = sortperm(x)
    cb = cumsum(bi[ordx]); btot = cb[n]
    g = zeros(n)
    k = 1
    while k <= n
        j = k
        @inbounds while j < n && x[ordx[j+1]] == x[ordx[k]]
            j += 1
        end
        # a run of tied x shares the symmetric subgradient
        lo = k - 1
        hi = n - j
        blo = k > 1 ? cb[k-1] : 0.0
        bhi = btot - cb[j]
        s = lo - hi
        @inbounds for t in k:j
            i = ordx[t]
            dS1 = 2 * (L[i] - R[i])
            dS2 = s * bi[i] + blo - bhi
            daa = 2 * s
            g[i] = (dS1 - 2 * dS2 / (n - 2) + daa * bb / ((n - 1) * (n - 2))) / (n * (n - 3))
        end
        k = j + 1
    end
    g
end


abstract type AbstractDcorCache end

"""
    DcorCache(ctrl::Vector{Float64}; wbar=1.0)

One control of length `n`: its fixed row sums and ranks, and per-round scratch.
"""
struct DcorCache <: AbstractDcorCache
    ctrl::Vector{Float64}      # standardised control
    bi::Vector{Float64}        # row sums of the control
    bb::Float64                # their total
    ry::Vector{Int}            # rank of each control value
    ordx::Vector{Int}          # scratch from here down
    L::Vector{Float64}
    R::Vector{Float64}
    cb::Vector{Float64}
    g::Vector{Float64}
    cnt::Fenwick
    sy::Fenwick
    wbar::Float64              # mean training weight
end

function DcorCache(ctrl::Vector{Float64}; wbar::Float64=1.0)
    n = length(ctrl)
    bi = _rowsums(ctrl)
    DcorCache(ctrl, bi, _ordered_sum(bi), invperm(sortperm(ctrl)),
        zeros(Int, n), zeros(n), zeros(n), zeros(n), zeros(n), Fenwick(n), Fenwick(n), wbar)
end

# `_signed_ydist(x, dc.ctrl)` into `dc.L` and `dc.R`.
function _signed_ydist!(dc::DcorCache, x::AbstractVector)
    n = length(x)
    y, ry, ordx = dc.ctrl, dc.ry, dc.ordx
    for (out, forward) in ((dc.L, true), (dc.R, false))
        fill!(dc.cnt.t, 0.0)
        fill!(dc.sy.t, 0.0)
        k = 1
        while k <= n
            j = k
            @inbounds while j < n && x[ordx[forward ? j + 1 : n - j]] == x[ordx[forward ? k : n - k + 1]]
                j += 1
            end
            @inbounds for t in k:j
                i = ordx[forward ? t : n - t + 1]
                r = ry[i]
                yi = y[i]
                c_lo = fsum(dc.cnt, r); s_lo = fsum(dc.sy, r)
                c_all = fsum(dc.cnt, n); s_all = fsum(dc.sy, n)
                out[i] = (c_lo * yi - s_lo) + ((s_all - s_lo) - (c_all - c_lo) * yi)
            end
            @inbounds for t in k:j
                i = ordx[forward ? t : n - t + 1]
                fadd!(dc.cnt, ry[i], 1.0)
                fadd!(dc.sy, ry[i], y[i])
            end
            k = j + 1
        end
    end
    return nothing
end

"""
    dcov2_grad!(dc::DcorCache, x::AbstractVector)

`dcov2_grad(x, dc.ctrl)` written to `dc.g`, a vector of length `n`, and returned.
"""
function dcov2_grad!(dc::DcorCache, x::AbstractVector)
    n = length(x)
    n == length(dc.ctrl) || throw(DimensionMismatch("prediction and control lengths differ."))
    n >= 4 || error("The unbiased distance covariance is undefined below 4 observations, got $n.")
    sortperm!(dc.ordx, x)
    _signed_ydist!(dc, x)
    bi, bb, ordx, cb, g = dc.bi, dc.bb, dc.ordx, dc.cb, dc.g
    @inbounds begin
        acc = 0.0
        for t in 1:n
            acc += bi[ordx[t]]
            cb[t] = acc
        end
    end
    btot = cb[n]
    k = 1
    while k <= n
        j = k
        @inbounds while j < n && x[ordx[j+1]] == x[ordx[k]]
            j += 1
        end
        lo = k - 1
        hi = n - j
        blo = k > 1 ? cb[k-1] : 0.0
        bhi = btot - cb[j]
        s = lo - hi
        @inbounds for t in k:j
            i = ordx[t]
            dS1 = 2 * (dc.L[i] - dc.R[i])
            dS2 = s * bi[i] + blo - bhi
            g[i] = (dS1 - 2 * dS2 / (n - 2) + 2 * s * bb / ((n - 1) * (n - 2))) / (n * (n - 3))
        end
        k = j + 1
    end
    return g
end

"""
    dcov2_value!(dc::DcorCache, x::AbstractVector)

`dcov2(x, dc.ctrl)` as `sum(g .* x)`, `g` from `dcov2_grad!`. Returns a scalar.
"""
function dcov2_value!(dc::DcorCache, x::AbstractVector)
    g = dcov2_grad!(dc, x)
    x0 = Float64(minimum(x))
    s = 0.0
    @inbounds for i in eachindex(g)
        s += g[i] * (Float64(x[i]) - x0)
    end
    return s
end

"""
    GroupedDcorCache(ctrl::Vector{Float64}, gi::GroupIndex; wbar=1.0, label=nothing)

One `DcorCache` per group of `gi`, the control standardised within the group. Groups below 4 rows,
or whose control has no usable spread, are skipped.
"""
struct GroupedDcorCache <: AbstractDcorCache
    caches::Vector{DcorCache}
    rows::Vector{Vector{Int}}
    g::Vector{Float64}
    wbar::Float64
end

function GroupedDcorCache(ctrl::Vector{Float64}, gi::GroupIndex; wbar::Float64=1.0, label=nothing)
    length(gi) == length(ctrl) ||
        throw(DimensionMismatch("control and group lengths differ."))
    caches = DcorCache[]
    rows = Vector{Int}[]
    for g in 1:ngroups(gi)
        r = Int.(group_rows(gi, g))
        length(r) >= 4 || continue
        c = ctrl[r]
        s = sort(c)
        # no usable spread: at most one value either side of a tie, or a spread at rounding level
        s[2] < s[end-1] || continue
        s[end] - s[1] > 64 * eps(max(1.0, abs(s[1]), abs(s[end]))) || continue
        m, sd = _ordered_mean_std(c)
        isfinite(sd) && sd > 0 || continue
        push!(caches, DcorCache((c .- m) ./ sd))
        push!(rows, r)
    end
    isempty(caches) && error(
        isnothing(label) ?
        "No group has at least 4 rows and a control with a usable spread, so there is nothing for " *
        "the within-group penalty to act on." :
        "No group has at least 4 rows and a usable spread in $label, so there is nothing for the " *
        "within-group penalty to act on with it."
    )
    GroupedDcorCache(caches, rows, zeros(length(ctrl)), wbar)
end

"""
    Controls(cols, weights, labels)

`k` standardised controls of length `n`, their `k` weights and `k` labels for error messages.
"""
struct Controls
    cols::Vector{Vector{Float64}}
    weights::Vector{Float64}
    labels::Vector{String}
end

"""
    SummedDcorCache(caches, weights, acc, rows, wbar)

`k` control caches whose gradients are summed with `weights` into `acc`, of length `n`. `rows` are
the rows any control acts on within groups.
"""
struct SummedDcorCache{C<:AbstractDcorCache} <: AbstractDcorCache
    caches::Vector{C}
    weights::Vector{Float64}
    acc::Vector{Float64}
    rows::Vector{Int}
    wbar::Float64
end

"""
    EvalCtrl{L,C}

The `k` controls of the eval set and the scratch of `eval_penalty!`. `dep` holds the `k`
dependences of the last call.
"""
struct EvalCtrl{L,C<:AbstractDcorCache}
    caches::Vector{C}
    weights::Vector{Float64}
    lambda::Float64
    penalized::Bool
    sign::Float64
    n::Int
    rows::Vector{Int}
    x::Vector{Float64}
    a::Vector{Float64}
    dep::Vector{Float64}
    d::Vector{Float64}
    da::Vector{Float64}
end

function eval_ctrl(params, ::Type{L}, K::Int, ctrl::Controls, group, feval, n::Int) where {L}
    within = hasproperty(params, :ctrl_within_group) && params.ctrl_within_group
    if within
        isnothing(group) && error(
            "`ctrl_within_group` is set but the eval set has no groups. Pass `eval_group_name` when " *
            "fitting from a table, or `group_eval` alongside `x_eval`.")
        caches = [GroupedDcorCache(c, group; label=l) for (c, l) in zip(ctrl.cols, ctrl.labels)]
    else
        caches = [DcorCache(c) for c in ctrl.cols]
    end
    lambda = hasproperty(params, :ctrl_lambda) ? params.ctrl_lambda : 0.0
    penalized = lambda > 0 && feval === _own_metric(L)
    nc = length(caches)
    return EvalCtrl{L,eltype(caches)}(caches, ctrl.weights, lambda, penalized,
        is_maximise(feval) ? -1.0 : 1.0, n, collect(ctrl_rows(L, K)), zeros(n), zeros(n),
        zeros(nc), zeros(nc), zeros(nc))
end

# `(d, da)`: one control's dependence, and the same weighted by `a` under `:gaussian_mle`
function _dependence(dc::DcorCache, x, a, n)
    d = dcov2_value!(dc, x)
    return d, isnothing(a) ? d : d * _ordered_sum(a) / length(a)
end
function _dependence(gc::GroupedDcorCache, x, a, n)
    dg = zeros(length(gc.caches))
    dag = zeros(length(gc.caches))
    @threads for k in eachindex(gc.caches)
        rows = gc.rows[k]
        dg[k] = length(rows) / n * dcov2_value!(gc.caches[k], view(x, rows))
        dag[k] = isnothing(a) ? dg[k] : dg[k] * _ordered_sum(view(a, rows)) / length(rows)
    end
    return _ordered_sum(dg), _ordered_sum(dag)
end

# pooled controls run in parallel; within groups `_dependence` threads over the groups
function _dependences!(ec::EvalCtrl{<:Any,DcorCache}, a)
    @threads for c in eachindex(ec.caches)
        ec.d[c], ec.da[c] = _dependence(ec.caches[c], ec.x, a, ec.n)
    end
end
function _dependences!(ec::EvalCtrl, a)
    for c in eachindex(ec.caches)
        ec.d[c], ec.da[c] = _dependence(ec.caches[c], ec.x, a, ec.n)
    end
end

"""
    eval_penalty!(ec::EvalCtrl, p::AbstractMatrix)

For predictions `p` of size `(K, n)`, the `k` dependences and the penalty added to the metric.
"""
function eval_penalty!(ec::EvalCtrl{L}, p::AbstractMatrix) where {L}
    ph = p isa Array ? p : Array(p)
    fill!(ec.dep, 0.0)
    term = 0.0
    for k in ec.rows
        @inbounds for i in eachindex(ec.x)
            ec.x[i] = _dep_transform(L, Float64(ph[k, i]))
        end
        a = nothing
        if L <: GaussianMLE
            @inbounds for i in eachindex(ec.a)
                ec.a[i] = exp(-2 * Float64(ph[k+1, i]))
            end
            a = ec.a
        end
        _dependences!(ec, a)
        for c in eachindex(ec.caches)
            ec.dep[c] += ec.d[c]
            term += ec.weights[c] * ec.da[c]
        end
    end
    ec.dep ./= length(ec.rows)
    return ec.dep, ec.sign * ec.lambda * _dep_coef(L) * term / length(ec.rows)
end
