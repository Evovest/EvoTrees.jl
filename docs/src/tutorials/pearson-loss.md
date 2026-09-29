# Per-date correlation

In a cross-sectional panel, such as stock returns by date and asset, a model is often judged
date by date: on each date the assets are ranked by the prediction, and what counts is how well
that ranking lines up with the realised returns. The usual score is the Pearson correlation
between prediction and target within each date, averaged over dates.

Squared error on the raw target is a poor fit for that goal. It is dominated by the dates with
the largest volatility, and it spends trees on the level each date shares across all assets,
which does nothing for a ranking within the date. `loss = :pearson` and `loss = :pearson_rank`
train on the per-date correlation instead.

## Getting started

```julia
using EvoTrees
using Random
using Statistics
using Printf
```

## A synthetic panel

Each row is an asset on a date, and each date holds between 200 and 400 assets. A date has its
own volatility and its own level, the return shared by all its assets. The level is partly
predictable from `z`, a feature that is the same for every asset on a date. Within a date, the
target carries a weak nonlinear signal in `x1` to `x4` plus Student-t noise with 3 degrees of
freedom, so a few assets on most dates have extreme returns.

```julia
rng = Xoshiro(123)
ndates = 400
sizes = rand(rng, 200:400, ndates)
date = reduce(vcat, [fill(d, n) for (d, n) in enumerate(sizes)])
nobs = length(date)
rows = [(s - n + 1):s for (s, n) in zip(cumsum(sizes), sizes)]

vol = 0.02 .* exp.(0.5 .* randn(rng, ndates))                  # volatility of each date
z = randn(rng, ndates)                                           # a date-level feature
level = 3 .* vol .* (0.8 .* z .+ 0.6 .* randn(rng, ndates))     # return shared by the date

x1, x2, x3, x4, x5 = (randn(rng, nobs) for _ in 1:5)
f = x1 .+ 0.7 .* sin.(2 .* x2) .+ 0.5 .* x3 .* x4               # the signal, x5 is noise
signal = similar(f)
for r in rows
    signal[r] = (f[r] .- mean(f[r])) ./ std(f[r]; corrected=false)
end

chi2 = randn(rng, nobs) .^ 2 .+ randn(rng, nobs) .^ 2 .+ randn(rng, nobs) .^ 2
noise = randn(rng, nobs) ./ sqrt.(chi2 ./ 3) ./ sqrt(3)          # Student-t(3), variance 1

rho = 0.1
y = level[date] .+ vol[date] .* (rho .* signal .+ sqrt(1 - rho^2) .* noise)
x = hcat(x1, x2, x3, x4, x5, z[date])
```

Dates are kept in order and split into train, validation and test periods, so every evaluation
is out of sample in time:

```julia
span(ds) = first(rows[first(ds)]):last(rows[last(ds)])
train_dates, valid_dates, test_dates = 1:240, 241:320, 321:400
train, valid, test = span(train_dates), span(valid_dates), span(test_dates)
```

## Measuring per-date correlation

The helper below computes the Pearson correlation within each date and averages over dates,
each date counting once. A date whose prediction is constant scores zero. On these data it is
the same quantity as `metric = :pearson` with unit weights; the metric also leaves out a date
whose target is constant, where this helper would give `NaN`.

```julia
safe_cor(a, b) = std(a) > 0 ? cor(a, b) : 0.0
date_cor(p, dates) = mean(safe_cor(p[rows[d]], y[rows[d]]) for d in dates)
```

For reference, the true signal itself scores on the test dates:

```julia
signal_r = date_cor(signal, test_dates)
```

## Three fits

The three models share one configuration and differ only in the loss. The groups are the dates:
`group_train` and `group_eval` take one date per row. `metric = :pearson` is the default for the
two correlation losses and is set here for `:mse` too, so all three stop early on the per-date
correlation of the validation dates. `metric = :pearson` needs `group_eval` (or
`eval_group_name` when fitting from a table), and is measured on the untransformed target for all
three losses.

```julia
config(loss) = EvoTreeRegressor(
    loss=loss,
    metric=:pearson,
    nrounds=3000,
    early_stopping_rounds=100,
    eta=0.05,
    max_depth=4,
    seed=1,
)

function fit_and_score(loss)
    m = EvoTrees.fit(config(loss);
        x_train=x[train, :], y_train=y[train], group_train=date[train],
        x_eval=x[valid, :], y_eval=y[valid], group_eval=date[valid],
        feature_names=["x1", "x2", "x3", "x4", "x5", "z"],
        verbosity=0)
    best_iter = m.info[:logger][:best_iter]
    p = zeros(nobs)
    p[test] = m(x[test, :]; ntree_limit=best_iter)
    return (; loss, m, best_iter, test_r=date_cor(p, test_dates))
end

res_mse = fit_and_score(:mse)
res_pearson = fit_and_score(:pearson)
res_rank = fit_and_score(:pearson_rank)

println("loss            best_iter   test r")
for r in (res_mse, res_pearson, res_rank)
    @printf("%-14s %10d %8.4f\n", r.loss, r.best_iter, r.test_r)
end
```

`best_iter` is the round with the best validation score, and the test predictions pass
`ntree_limit = best_iter` so that they come from that round.

| **Loss**        | **Best iteration** | **Test per-date r** |
|-----------------|--------------------|---------------------|
| `:mse`          | 125                | 0.0783              |
| `:pearson`      | 136                | 0.0994              |
| `:pearson_rank` | 138                | 0.1020              |

The true signal scores 0.1123 on the same dates, roughly the ceiling for any
model of these features. `:mse` on the raw target trails the correlation losses: its fit is
dominated by the most volatile dates, and it can spend splits on `z` to fit the date levels,
which are worthless for a ranking within a date. `:pearson_rank` scores a little above
`:pearson` here, a gap within the noise of 80 test dates.

## What the loss fits

Under `:pearson`, each date's prediction, centred on its own mean, is fitted by weighted squared
error to the date's standardised target. Within a date that error equals `(s - r)^2 + 1 - r^2`,
for prediction spread `s` and Pearson correlation `r`, so it raises the correlation while holding
the spread near `max(r, 0)`. Each row counts once, so a date weighs by its number of rows.
`:pearson_rank` fits the same objective to the standard normal quantile of the target's rank
within its date instead of the standardised target; the ranks are unweighted. It depends only on
the order of the target within a date, so a few extreme returns do not dominate a date's fit, and
it tends to do better when the target has heavy tails. On a target already replaced within each
date by the normal quantiles of its ranks the two agree exactly; on plain ranks they differ
slightly. Either loss is close to `:mse` on a target you standardise or rank-transform within each
date yourself. What the loss adds is doing that inside the fit, from the same groups the metric
uses, and centring each date's prediction on its own mean, so a feature that is constant within a
date, such as `z`, cannot on its own reduce the loss, though it can still enter through an
interaction.

The predictions are scores within a date. Their level carries no meaning across dates, and their
spread is on the scale of the correlation, not of the target. Rank or standardise them
within each date before combining them with other signals or turning them into positions.

An offset is part of that score. The loss centres the offset plus the trees within each date and
holds their spread near the correlation, so an offset that is constant within a date has no
effect, and one on the target's scale has far more spread than the loss keeps, which the trees
then mostly work to shrink. Give it on the scale of a `:pearson` model's output, such as a
previous `:pearson` model's predictions.

To count dates equally in training rather than by size, weight each row by `nbar / n_g`, with
`n_g` the size of its date and `nbar` the mean size, and pass it as `w_train` (or as a
`weight_name` column when fitting from a table). Keep the evaluation weights at 1, so that the
metric still counts dates equally:

```julia
w_train = mean(sizes[train_dates]) ./ sizes[date[train]]
```

Both losses need groups, so like `:lambdarank` they are available through `EvoTrees.fit` only,
not through the MLJ interface.
