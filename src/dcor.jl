# Univariate distance covariance in O(n log n) (Chaudhuri & Hu 2019), with an analytic
# gradient at the same cost. Used as a decorrelation penalty: a control variable the
# predictions should carry no dependence on. The direct definition is O(n^2) in time and
# memory, which rules it out as a per-round penalty at ordinary training sizes.

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

# Sums in index order. `sum`, `mean` and `std` let the compiler split an array across SIMD lanes,
# whose width depends on the CPU, so the control's scaling, and after enough rounds the trees, would
# differ in their last bits between machines. The penalty's per-round sums already run in order.
function _ordered_sum(x)
    s = 0.0
    @inbounds for v in x
        s += Float64(v)
    end
    return s
end

# Mean and sample standard deviation, both summed in index order, see `_ordered_sum`. The sums are
# taken from the first value, so a control on an absolute scale, a timestamp or a price, keeps its
# digits: summed as it stands, the running total of a column near 1e12 rounds at its own scale, and
# the spread moves with it, enough to flip a split. The mean itself is then rounded at the column's
# scale, which moves every centred value alike and so leaves the distance statistics unchanged.
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

# a_i. = sum_j |x_i - x_j| for every i
function _rowsums(x::AbstractVector)
    n = length(x)
    ord = sortperm(x)
    # every term below is a sum of differences, so the vector is shifted to its own smallest
    # value first. Without that the prefix sums carry the offset and the leading digits cancel:
    # a control on an absolute scale, a timestamp or a price, loses most of its precision.
    x0 = Float64(x[ord[1]])
    xs = [Float64(x[i]) - x0 for i in ord]
    cs = cumsum(xs)
    out = zeros(n)
    @inbounds for (k, i) in enumerate(ord)
        out[i] = (k * xs[k] - cs[k]) + ((cs[n] - cs[k]) - (n - k) * xs[k])
    end
    out
end

# For each i: L_i = sum over {j : x_j < x_i} of |y_i - y_j|, R_i the same over x_j > x_i.
# Fenwick trees over the rank of y hold counts and sums of y for the js seen so far.
function _signed_ydist(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    y0 = Float64(minimum(y))          # shift-invariant, see `_rowsums`
    L = zeros(n)
    R = zeros(n)
    for (out, order) in ((L, ordx), (R, reverse(ordx)))
        cnt = Fenwick(n)
        sy = Fenwick(n)
        k = 1
        while k <= n
            # the run of equal `x` starting at k: ties belong to neither side, so the whole
            # run is scored against what came before it and only then inserted
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

# sum_ij |x_i - x_j| |y_i - y_j|
function _cross_term(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    cnt = Fenwick(n); sy = Fenwick(n); sx = Fenwick(n); sxy = Fenwick(n)
    x0 = Float64(minimum(x)); y0 = Float64(minimum(y))   # shift-invariant, see `_rowsums`
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
    dcov2(x, y)

Unbiased squared distance covariance of two real vectors (Szekely & Rizzo 2014), in O(n log n).
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
    dcor2(x, y)

Squared distance correlation, `dcov2(x, y) / sqrt(dcov2(x, x) * dcov2(y, y))`.
"""
function dcor2(x::AbstractVector, y::AbstractVector)
    # each variance is a sum of squares over n(n - 3), so it is >= 0 in exact arithmetic, but the
    # O(n log n) form can round marginally below zero for nearly constant input; testing each on its
    # own rejects the case where both round negative, which a single `vx * vy > 0` would let through
    vx = dcov2(x, x)
    vy = dcov2(y, y)
    (vx > 0 && vy > 0) || return 0.0
    return dcov2(x, y) / sqrt(vx * vy)
end

"""
    dcov2_grad(x, y)

Gradient of `dcov2(x, y)` with respect to `x`, holding `y` fixed, in O(n log n). `dcov2` is
piecewise linear in `x`, so its Hessian is zero almost everywhere and is not returned.
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
        # counts and `b` masses strictly either side of this run. Boosted predictions are
        # heavily tied, so resolving ties by index order here would make the gradient a
        # function of row order; the symmetric subgradient is used instead.
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
    DcorCache(ctrl)

Everything the penalty needs that does not change between rounds, plus the scratch it reuses.

The control is fixed at initialisation, so its row sums, their total, and the ranks used by the
Fenwick sweeps are computed once here rather than on every round. What is left per round is one
`sortperm` of the predictions and the sweeps themselves, writing into buffers held here. The
allocating `dcov2_grad` stays as the one-shot entry point; training goes through this.
"""
struct DcorCache <: AbstractDcorCache
    ctrl::Vector{Float64}      # standardised control
    bi::Vector{Float64}        # row sums of the control, fixed
    bb::Float64                # their total, fixed
    ry::Vector{Int}            # rank of each control value, fixed
    ordx::Vector{Int}          # scratch from here down
    L::Vector{Float64}
    R::Vector{Float64}
    cb::Vector{Float64}
    g::Vector{Float64}
    cnt::Fenwick
    sy::Fenwick
    wbar::Float64              # mean training weight, see `_penalize_row!`
end

function DcorCache(ctrl::Vector{Float64}; wbar::Float64=1.0)
    n = length(ctrl)
    bi = _rowsums(ctrl)
    DcorCache(ctrl, bi, _ordered_sum(bi), invperm(sortperm(ctrl)),
        zeros(Int, n), zeros(n), zeros(n), zeros(n), zeros(n), Fenwick(n), Fenwick(n), wbar)
end

# `_signed_ydist` against a fixed control, writing into the cache and reusing its Fenwick trees.
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
    dcov2_grad!(dc::DcorCache, x)

`dcov2_grad(x, dc.ctrl)` into `dc.g`, reusing the cache's buffers and its control-side work.
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
    GroupedDcorCache(ctrl, gi::GroupIndex)

One `DcorCache` per group, so the penalty acts on the dependence within each group rather than
pooled over the sample. On a panel where a group is a date and a row is an asset, the pooled
statistic is dominated by the date-level component of the control, while the exposure a portfolio
carries is the cross-sectional one. Each group's control is standardised on its own rows, so a
date with a wider spread does not weigh more. Groups below 4 rows are skipped, as are groups whose
control has no usable spread: constant up to the rounding of the standardised control, or every
value tied but at most one either side of them. The unbiased distance covariance is undefined or zero there.
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
        # every value tied but at most one either side makes the distance variance exactly zero;
        # a spread at the rounding level of the globally standardised control is noise, and one
        # that underflows would scale to NaN
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

The control variables of a fit, each validated and standardised by `build_ctrl`, with the weight
each one's penalty term is multiplied by and the name an error refers to it by.
"""
struct Controls
    cols::Vector{Vector{Float64}}
    weights::Vector{Float64}
    labels::Vector{String}
end

"""
    SummedDcorCache(caches, weights, acc, rows, wbar)

Several controls at once. Each control keeps its own cache, over the whole sample or within groups,
and the penalty is the sum of their terms, each times its weight. It acts on each control's
own dependence, so a dependence that shows only in a combination of controls is not seen.

The sum runs in the order the controls were given and starts from the first control's term rather
than from zero, so one control of weight 1 gives exactly the gradient of its own cache, and the result
does not depend on how the controls' sweeps are spread over threads. `rows` lists the rows any control
acts on, which within groups leaves out the groups every control skips.
"""
struct SummedDcorCache{C<:AbstractDcorCache} <: AbstractDcorCache
    caches::Vector{C}
    weights::Vector{Float64}
    acc::Vector{Float64}
    rows::Vector{Int}
    wbar::Float64
end
