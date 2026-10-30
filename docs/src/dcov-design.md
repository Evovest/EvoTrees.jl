# Decorrelation penalty: design

This page describes how `ctrl_lambda` is defined and computed, and why. For usage, see the
[decorrelation tutorial](tutorials/decorrelation.md).

Notation: `n` training rows with weights `w_i` and total weight `W`; `p` the raw predictions (on
the link scale); controls `c_1, ..., c_k`, each centred and scaled to unit standard deviation;
`λ = ctrl_lambda` and `ω_j` the entries of `ctrl_weights` (1 by default).

## Statistic

Dependence is measured by the unbiased squared distance covariance `dcov2(p, c)` (Székely and
Rizzo, 2014). In population it is zero only when `p` and `c` are independent, so it also captures
nonlinear dependence. It is the same for `p` and `-p`, so the penalty pushes towards independence,
not towards a negative correlation.

A Pearson correlation would cost O(n) rather than O(n log n), but only sees linear dependence: when
the target depends on the square of a symmetric control, a fitted model has almost no correlation
with it and a Pearson penalty has nothing to remove. Distance correlation costs the same as
`dcov2`, but its gradient is inversely proportional to the spread of `p`, so it pushes hardest in
the first rounds, when predictions are nearly flat, and its weight scales with the square of the
target. The gradient of `dcov2` depends on `p` only through its order, so its push is steady.

The estimator is a U-statistic: unbiased for a population value that is never negative, but a
sample value can fall below zero near independence. It needs at least 4 rows. A control with at
most one value either side of a tie has zero distance variance and is rejected, or skipped within
a group.

The estimate is kept unbiased, with no floor at zero. Near independence it is negative about half
the time: with 150 rows per group, the mean of `dcor2` over 75 independent groups has a standard
deviation of about 0.0011 around zero.

Two alternatives were tried. Clipping each group's estimate at zero, `max(dcov2_g, 0)`, switched off
54 to 60% of the groups in each round at a moderate weight, left more held-out dependence wherever
the fit settled, and made 3 to 12% of the groups switch on or off from one round to the next. The
V-statistic, which is never negative, shrank the predictions and lowered their correlation with the
target in every run. With the unbiased estimate, very large weights still inflate the predictions
(see Step size below), so the weight is best tuned on held-out data.

## Objective

Under `:mse` the training objective is

```
sum_i w_i (y_i - p_i)^2 + λ W sum_j ω_j dcov2(p, c_j)
```

that is, `W` times the sum of the weighted mean squared error and `λ sum_j ω_j dcov2(p, c_j)`. The
factor `W` puts the penalty, whose gradient is O(1/n) per row, on the scale of the loss, which sums over rows,
so `λ` depends neither on `n` nor on the scale of the weights. Controls are standardised, so `λ`
does not depend on their units. `dcov2` is homogeneous of degree one in `p` and the squared error
of degree two, so `λ` scales with the target: a target 10 times larger needs 10 times the weight.

`dcov2` is piecewise linear in `p`, so its Hessian is zero almost everywhere. The penalty adds
`λ W ∂dcov2/∂p_i` to the gradient of row `i` and leaves the Hessian unchanged.

The penalty acts on the full prediction, offset included, in training and in evaluation, since that
is the prediction that is scored and the one the evaluation metric measures. Under `:mse`, fitting
`y - offset` without an offset penalises the trees' part alone.

### Curvature weighting

Under other losses the penalty gradient of row `i` is multiplied by `h_i / (2 w_i)`, the loss's
Hessian relative to that of `:mse`. A leaf moves by its gradient sum over its Hessian sum, so with
this weighting a given `λ` moves leaves about as much as under `:mse`. Without it the penalty
dominates rows where the likelihood is flat, such as `:logloss` rows predicted near 0 or 1, and the
fit diverges.

Since the gradient `g` of `dcov2(p, c)` depends on `p` only through its order, `g_i φ'(p_i)` is the
gradient of `dcov2(φ(p), c)` for any increasing `φ`, and each loss's curvature is such a
derivative. The penalty term per control, in the units of the loss's own metric, is therefore:

| Loss            | Penalty term                                          | Exact                   |
|:----------------|:------------------------------------------------------|:------------------------|
| `:mse`          | `λ dcov2(p, c)`                                       | yes                     |
| `:logloss`      | `λ / 2 * dcov2(sigmoid(p), c)`                        | yes                     |
| `:poisson`      | `λ dcov2(exp(p), c)`                                  | yes                     |
| `:gamma`        | `λ dcov2(p, c)`                                       | at a calibrated fit     |
| `:tweedie`      | `λ / (2 - ρ) * dcov2(exp((2 - ρ) p), c)`, `ρ = 1.5`   | at a calibrated fit     |
| `:gaussian_mle` | `λ / 2 * mean(σ^-2) * dcov2(μ, c)`                    | at a constant scale `σ` |
| `:pearson`      | `λ dcov2(p, c)`, within groups only                   | on the groups it scores |

Under `:gaussian_mle` only the location `μ` is penalised. In the code `φ` is `_dep_transform` and
the coefficient `_dep_coef`.

## Pooled and within groups

By default one `dcov2` is computed over all rows. With `ctrl_within_group = true`, group `g` of
`group_name` / `group_train` adds `λ W_g dcov2(p_g, c_g)`, with `W_g` the total training weight of
its rows (its number of rows without weights) and the control standardised within the group. Under
`:mse` the objective is

```
sum_i w_i (y_i - p_i)^2 + λ sum_j ω_j sum_g W_g dcov2(p_g, c_jg)
```

with `p_g` and `c_jg` the predictions and control `j` on the rows of group `g`. Each group's penalty
carries the weight its rows carry in the loss, and `dcov2` itself is unweighted within a group. With
the mean weight times the number of rows in place of `W_g`, the penalty per unit of loss weight
would vary between groups with their mean weight. Groups with fewer than 4 rows, or whose control
has no usable spread, are skipped.

This targets the dependence within each group: differences in the control's level or spread
between groups do not count. Under `:mse` the gradient sums to zero within each group, since
`dcov2` does not change when a constant is added to `p_g`. The trees can still move group means
through features that vary between groups, and the loss then corrects those moves. Groups run in
parallel.

## Several controls

With several controls the penalty is `sum_j ω_j dcov2(p, c_j)`, not the distance covariance of `p`
with the vector of controls. The O(n log n) algorithm below holds for two univariate variables
only, and a joint distance would mix the controls' scales, whereas the sum makes each control's
weight explicit. A dependence visible only jointly is not penalised, for example
`p = sign(c_1) sign(c_2)` with independent, symmetric controls. Pooled controls run in parallel.

## Algorithm

With `a_ij = |p_i - p_j|`, `b_ij = |c_i - c_j|`, row sums `a_i.`, `b_i.` and totals `a..`, `b..`:

```
dcov2 = (S1 - 2 S2 / (n - 2) + a.. b.. / ((n - 1)(n - 2))) / (n (n - 3))
S1 = sum_ij a_ij b_ij,   S2 = sum_i a_i. b_i.
```

Row sums come from a sort and prefix sums. `S1` comes from a sweep over the rows in the order of
`p`, with Fenwick trees indexed by the rank of `c` that hold counts and sums over the rows already
visited (Chaudhuri and Hu, 2019). The gradient has the same structure:
`∂S1/∂p_i = 2 sum_j sign(p_i - p_j) b_ij` comes from a forward and a backward sweep, and the other
terms need only the number of rows below and above `p_i` and their `b` sums. Each round costs, per
control, one sort of `p` and two sweeps: O(n log n) time and O(n) memory instead of O(n²). The
control's row sums and ranks are computed once per fit.

Boosted predictions are heavily tied. Tied rows count on neither side, the symmetric subgradient of
`|p_i - p_j|` at zero, so the gradient does not depend on row order. Since `dcov2` is homogeneous
of degree one and `g` depends only on the order of `p`, `dcov2(p, c) = sum_i g_i p_i`, which gives
the value from the gradient.

Controls and the penalty's sums are held in `Float64`: in `Float32` a control on a large scale
loses its resolution (the step is 128 at a Unix timestamp). Sums run in index order and from a
shifted origin, so fits agree across CPUs and keep their precision when values are large.

At a million rows on 8 threads, one pooled control took about 0.56 s per round, its sweeps being
sequential, against 0.06 s for growing the tree. Within 1000 groups of 1000 rows, each control took
about 0.025 s per round.

## Evaluation metric

With evaluation data, `λ > 0` and the loss's own metric (the default), the logged metric adds the
penalty term of the table above, computed on the evaluation set with its controls standardised
there (within each evaluation group under `ctrl_within_group`, each group's term weighted by
`W_g / W` from the evaluation weights). Under `:mse` the metric is then the training objective on
the evaluation set divided by its total weight. For a maximised metric such as `:gaussian_mle` the
term is subtracted. Early stopping then follows the objective that training minimises, rather than
the base metric, which ignores the dependence. Other metrics have no exchange rate with `λ` and are
reported unchanged.

The logger also keeps `:base_metrics`, the metric without the penalty, and `:ctrl_dependence`, each
control's `dcov2(φ(p), c_j)` before `λ` and the weights. The callback adds what `eval_penalty!`
returns and holds no loss-specific code.

## `:pearson`

`loss = :pearson` fits each group's centred prediction to its standardised target, with curvature
`2w` as under `:mse`, so group `g` adds `λ W_g ∂dcov2_g/∂p` to the gradient. Groups of one row or
with a constant target are not scored by the loss, have no curvature and get no penalty. The loss
does not change when a constant is added to a group's predictions, so a pooled penalty would move
group levels that nothing holds in place: `:pearson` takes the penalty within groups only. The
target is standardised within each group, so `λ` does not depend on its scale.

The default metric, `:pearson`, is a correlation with no exchange rate with `λ`, so it is reported
without the penalty. `:pearson_mse`, the loss's objective per unit weight, is the loss's own metric,
so with `metric = :pearson_mse` the logged value adds the penalty. An evaluation group with a
constant target is left out of `:pearson_mse` but its dependence still counts in the penalty.

Under a strong penalty, early stopping on `:pearson` keeps the first tree. That tree is fitted
before the penalty acts, and the correlation falls in later rounds as the penalty removes the
control's share of the signal. On 100 groups of 50 rows whose control carried a third of each
group's signal, at `λ = 10`, it stopped at round 1 in 21 of 21 seeds. On `:pearson_mse` it ran 26 to
62 rounds, to at most 0.11 of round 1's within-group `dcor2`, at a per-group correlation of 0.729 to
0.793 against 0.845 to 0.875. At `λ = 1`, or with a control unrelated to the target, both metrics
reached the same per-group correlation within 0.001.

## GPU

The sort and the sweeps do not map onto a per-row kernel, so on GPU the penalty runs on the host:
each round the penalised gradient rows, their Hessian rows, the weights and the predictions are
copied to the host and the penalised gradient rows copied back. At a million rows on an A10G this
took about 0.34 s per round pooled and 0.026 s within 1000 groups, against 0.011 s for the trees.

## Step size

The penalty adds nothing to the Hessian, and its gradient depends on the order of `p`, not on its
spread, so its step does not shrink as the predictions flatten. Past some `λ * eta`, one round
carries the fit past independence and the next pushes it back, and the fit oscillates instead of
settling. In synthetic runs with predictions of standard deviation about 1, the pooled form started
oscillating at a `λ * eta` between 3 and 10, and the within-group form at 30 or more. The
correlation with the control swung by up to ±0.3 under `:mse` and ±0.8 under `:logloss`, never
reaching -1. A larger weight needs a smaller `eta`.

Without an offset the first round carries no penalty, since tied predictions give a zero gradient.
The second round applies the full push to nearly flat predictions and can overshoot to a negative
correlation, which later rounds correct.

At very large weights the predictions inflate, through the oscillation and through negative
estimates: once `dcov2` is below zero, scaling the predictions up lowers the penalty. Within groups
at `λ = 10000` and `eta = 0.01`, the per-group standard deviation of the predictions grew from 0.97
without a penalty to 13.9, with the training `dcov2` below zero.
