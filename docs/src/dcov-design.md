# Decorrelation penalty: design

This page describes how `ctrl_lambda` is defined and computed. For usage, see the
[decorrelation tutorial](tutorials/decorrelation.md).

Notation: `n` training rows with weights `w_i` and total weight `W`; `p` the raw predictions (on
the link scale); controls `c_1, ..., c_k`, each centred and scaled to unit standard deviation;
`λ = ctrl_lambda` and `ω_j` the entries of `ctrl_weights` (1 by default).

## Statistic

Dependence is measured by the unbiased squared distance covariance `dcov2(p, c)` (Székely and
Rizzo, 2014), which is zero in population only under independence. It is the same for `p` and
`-p`, so the penalty pushes towards independence rather than a negative correlation.

Pearson correlation is cheaper (O(n)) but only captures linear dependence. Distance covariance also
captures nonlinear dependence, at a cost of O(n log n). Distance correlation costs the same, but its
gradient is inversely proportional to the spread of `p`, so it pushes hardest when the predictions
are nearly flat. The gradient of `dcov2` depends on `p` only through its order, so its push is
steady.

The estimator is a U-statistic and is not floored at zero, so near independence it is negative
about half the time. It needs at least 4 rows. A control with at most one value either side of a
tie has zero distance variance and is rejected, or skipped within a group.

We considered two alternatives: clipping negative estimates at zero and using the V-statistic.
Clipping caused groups to alternate between active and inactive penalties, while the V-statistic
shrank predictions and reduced their correlation with the target. We therefore retain the unbiased
estimator. Very large weights still inflate the predictions (see Step size below), so tune the
weight on held-out data.

## Objective

Under `:mse` the training objective is

```
sum_i w_i (y_i - p_i)^2 + λ W sum_j ω_j dcov2(p, c_j)
```

The factor `W` puts the penalty on the scale of the summed loss, so `λ` depends neither on `n` nor
on the scale of the weights, and the standardised controls make it independent of their units. It
does scale with the target: `dcov2` is homogeneous of degree one in `p` and the squared error of
degree two, so a target 10 times larger needs 10 times the weight.

`dcov2` is piecewise linear in `p`, so the penalty adds `λ W ∂dcov2/∂p_i` to the gradient of row `i`
and nothing to the Hessian.

The penalty acts on the full prediction, offset included, in training and in evaluation. Under
`:mse`, fitting `y - offset` without an offset penalises the trees' part alone.

### Curvature weighting

Under other losses the penalty gradient of row `i` is multiplied by `h_i / (2 w_i)`, the loss's
Hessian relative to that of `:mse`, so a given `λ` moves leaves about as much as under `:mse`.
Without it the penalty dominates rows where the likelihood is flat, such as `:logloss` rows
predicted near 0 or 1, and the fit diverges.

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
carries the weight its rows carry in the loss, and `dcov2` is unweighted within a group. Groups with
fewer than 4 rows, or whose control has no usable spread, are skipped.

Only the dependence within groups counts, not differences in the control's level or spread between
groups. Under `:mse` the penalty gradient sums to zero within each group, since `dcov2` does not
change when a constant is added to `p_g`. The trees can still move group means through features that
vary between groups, and the loss corrects those moves. Groups run in parallel.

## Several controls

With several controls the penalty is `sum_j ω_j dcov2(p, c_j)`, not the distance covariance of `p`
with the vector of controls: the O(n log n) algorithm below holds for two univariate variables
only, and the sum makes each control's weight explicit. A dependence visible only jointly is not
penalised, for example `p = sign(c_1) sign(c_2)` with independent, symmetric controls. Pooled
controls run in parallel.

## Algorithm

With `a_ij = |p_i - p_j|`, `b_ij = |c_i - c_j|`, row sums `a_i.`, `b_i.` and totals `a..`, `b..`:

```
dcov2 = (S1 - 2 S2 / (n - 2) + a.. b.. / ((n - 1)(n - 2))) / (n (n - 3))
S1 = sum_ij a_ij b_ij,   S2 = sum_i a_i. b_i.
```

Row sums come from a sort and prefix sums. `S1` comes from a sweep over the rows in the order of
`p`, with Fenwick trees indexed by the rank of `c` holding the counts and sums of the rows already
visited (Chaudhuri and Hu, 2019). For the gradient, `∂S1/∂p_i = 2 sum_j sign(p_i - p_j) b_ij` takes
a forward and a backward sweep, and the other terms need only the number of rows below and above
`p_i` and their `b` sums. Per control and round this is one sort of `p` and two sweeps: O(n log n)
time and O(n) memory instead of O(n²). The control's row sums and ranks are computed once per fit.

Tied predictions, common in boosting, count on neither side (the symmetric subgradient of
`|p_i - p_j|` at zero), so the gradient does not depend on row order. Since `dcov2` is homogeneous
of degree one and `g` depends only on the order of `p`, `dcov2(p, c) = sum_i g_i p_i`, which gives
the value from the gradient.

Controls and the penalty's sums are held in `Float64`, since in `Float32` a control on a large
scale loses its resolution (the step is 128 at a Unix timestamp). Sums run in index order from a
shifted origin, so fits agree across CPUs and keep their precision at large values.

At a million rows on 8 threads, one pooled control took about 0.56 s per round (its sweeps are
sequential), against 0.06 s to grow the tree. Within 1000 groups of 1000 rows, each control took
about 0.025 s per round.

## Evaluation metric

With evaluation data, `λ > 0` and the loss's own metric (the default), the logged metric adds the
penalty term of the table above, so early stopping follows the penalised objective. The term is
computed on the evaluation set with its controls standardised there (within each evaluation group
under `ctrl_within_group`, each group's term weighted by `W_g / W` from the evaluation weights).
Under `:mse` the metric is then the training objective on the evaluation set divided by its total
weight. For a maximised metric such as `:gaussian_mle` the term is subtracted. Other metrics have no
exchange rate with `λ` and are reported unchanged.

The logger also keeps `:base_metrics`, the metric without the penalty, and `:ctrl_dependence`, each
control's `dcov2(φ(p), c_j)` before `λ` and the weights. The callback adds what `eval_penalty!`
returns and holds no loss-specific code.

## `:pearson`

`loss = :pearson` fits each group's centred prediction to its standardised target, with curvature
`2w` as under `:mse`, so group `g` adds `λ W_g ∂dcov2_g/∂p` to the gradient. Groups of one row or
with a constant target are not scored by the loss, have no curvature and get no penalty. The loss
does not change when a constant is added to a group's predictions, so a pooled penalty would move
group levels that nothing holds in place, and `:pearson` takes the penalty within groups only. The
target is standardised within each group, so `λ` does not depend on its scale.

The default metric, `:pearson`, has no exchange rate with `λ` and is reported without the penalty.
`:pearson_mse`, the loss's objective per unit weight, is its own metric, so with
`metric = :pearson_mse` the logged value adds the penalty. An evaluation group with a constant
target is left out of `:pearson_mse`, but its dependence still counts in the penalty.

Under a strong penalty, early stopping on `:pearson` keeps the first tree, which is fitted before
the penalty acts: the correlation then falls as the penalty removes the control's share of the
signal. Early stopping on `:pearson_mse` follows the penalised objective instead.

## GPU

The sort and the sweeps do not map onto a per-row kernel, so on GPU the penalty runs on the host.
Each round the penalised gradient rows, their Hessian rows, the weights and the predictions are
copied to the host, and the penalised gradient rows back. At a million rows on an A10G this took
about 0.34 s per round pooled and 0.026 s within 1000 groups, against 0.011 s for the trees.

## Step size

The penalty adds nothing to the Hessian and its gradient depends on the order of `p`, not its
spread, so its step does not shrink as the predictions flatten. Past some `λ * eta`, each round
overshoots independence and the fit oscillates instead of settling. In synthetic runs with
predictions of standard deviation about 1, this started at a `λ * eta` between 3 and 10 pooled and
at 30 or more within groups, with the correlation to the control swinging by up to ±0.3 under
`:mse` and ±0.8 under `:logloss`. A larger weight needs a smaller `eta`.

Without an offset the first round carries no penalty, since tied predictions give a zero gradient.
The second round applies the full push to nearly flat predictions and can overshoot to a negative
correlation, which later rounds correct.

At very large weights the predictions inflate, through the oscillation and through negative
estimates: once `dcov2` is below zero, scaling the predictions up lowers the penalty. Within groups
at `λ = 10000` and `eta = 0.01`, the per-group standard deviation of the predictions grew from 0.97
without a penalty to 13.9.
