# Pearson loss: design

This page describes how `loss = :pearson` is defined and computed. For usage, see the
[Pearson loss tutorial](tutorials/pearson-loss.md).

Notation: for a group `g` with `n_g` rows `i`, weights `w_i`, total weight `W_g` and
`a_i = w_i / W_g`; `p` the predictions and `p̄_g = sum_i a_i p_i`; `y` the target, standardised
within the group as `ỹ_i = (y_i - ȳ_g) / s_g`, with `ȳ_g` and `s_g` its weighted mean and standard
deviation.

## Objective

The correlation itself is a poor Newton objective: its Hessian scales with the correlation, so it
is near zero at low correlations. Instead, each group's centred prediction is fitted to its
standardised target by weighted squared error, as in CatBoost's QueryRMSE:

```
L = sum_g sum_i w_i (p_i - p̄_g - ỹ_i)^2
g_i = 2 w_i (p_i - p̄_g - ỹ_i)
h_i = 2 w_i
```

The gradient is exact, since `sum_i w_i (p_i - p̄_g - ỹ_i) = 0` within each group. `h_i` is the
exact diagonal without the centring term `-2 w_i^2 / W_g`, so it bounds the curvature from above.

With `s` the weighted standard deviation of `p` within the group and `r` its weighted correlation
with `y`,

```
sum_i a_i (p_i - p̄_g - ỹ_i)^2 = (s - r)^2 + 1 - r^2
```

For a given direction of the predictions, a group's term is smallest at `s = r` when `r > 0`, where
it equals `1 - r^2`, so lowering the loss raises the correlation.

## Groups and weights

Each group's term weighs by its total weight `W_g`, so with unit weights a group counts in
proportion to its size. Weights `w_i = 1 / n_g` make every group count the same. A group with fewer
than two rows or a constant target has `g_i = h_i = 0`, and fitting fails if no group is left.

A constant added to a group's predictions changes neither the loss nor the correlation, so no bias
is fitted.
