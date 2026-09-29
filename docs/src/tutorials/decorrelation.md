# Decorrelation penalty

A model can fit the target well and still owe much of its accuracy to one variable you do not
want it to depend on. In a stock return panel this is typically a risk exposure such as market
beta: the target loads on it, several features proxy it, and a model trained on squared error
picks it up. `ctrl_lambda` adds a penalty on the dependence between the prediction and such a
control variable, so the trees are pushed towards the part of the signal the control does not
explain.

The dependence is measured by the unbiased squared distance covariance `dcov2` of the prediction
and the control. Unlike a Pearson correlation, its population value is zero only under
independence, so a nonlinear dependence on the control is penalised as well as a linear one.

This tutorial builds a synthetic date by asset panel, fits it without a penalty, with the
penalty pooled over the whole sample, and with the penalty applied within each date, and then
shows how to choose `ctrl_lambda`.

## Getting started

```julia
using EvoTrees
using Random
using Statistics
using Printf
```

## A synthetic panel

Each row is an asset on a date. The control is the asset's market beta, which has a level
shared by the whole universe on a date plus the asset's own exposure. The target is the
asset's return: its beta times the market's return on that date, plus a part a model is meant
to find, `alpha`, plus noise. The market's return is positive on average, so over the training
period a model is rewarded for favouring high beta assets.

```julia
rng = Xoshiro(42)
ndates, nassets = 400, 150
nobs = ndates * nassets
date = repeat(1:ndates, inner=nassets)

beta = 1.0 .+ 0.3 .* randn(rng, ndates)[date] .+ 0.4 .* randn(rng, nobs)
mkt = 0.5 .+ randn(rng, ndates)
alpha = randn(rng, nobs)
y = beta .* mkt[date] .+ 0.5 .* alpha .+ randn(rng, nobs)

x = hcat(
    alpha .+ 0.5 .* randn(rng, nobs),                         # a noisy view of alpha
    beta .+ 0.2 .* randn(rng, nobs),                          # volatility, a proxy for beta
    0.5 .* alpha .+ 0.5 .* beta .+ 0.5 .* randn(rng, nobs),   # a feature mixing both
    randn(rng, nobs),                                         # noise
)
```

Dates are kept in order and split into train, validation and test periods, so every
evaluation is out of sample in time:

```julia
rows(d) = ((d - 1) * nassets + 1):(d * nassets)
span(dates) = first(rows(first(dates))):last(rows(last(dates)))

train_dates, valid_dates, test_dates = 1:250, 251:325, 326:400
train = span(train_dates)

x_train, y_train = x[train, :], y[train]
beta_train, date_train = beta[train], date[train]
```

## Measuring dependence per date

A cross-sectional model is judged date by date, so every quantity below is computed within
each date and averaged over dates:

- `corr`: the Pearson correlation of the prediction with the target;
- `corr_neutral`: the same against the beta-neutral target, the residual of the target after
  regressing it on beta within the date. This is the return a beta-hedged portfolio earns;
- `dep`: the squared distance correlation of the prediction with beta. It is bias corrected,
  so it sits near zero under independence and can come out slightly negative;
- `dep_pooled`: the same statistic over all rows of the period at once, dates mixed.

`EvoTrees.dcor2` computes the squared distance correlation in `O(n log n)`. It is internal,
hence not exported.

```julia
function neutralise(v, c)
    out = similar(v)
    for d in 1:ndates
        r = rows(d)
        vd = v[r] .- mean(v[r])
        cc = c[r] .- mean(c[r])
        out[r] = vd .- (sum(vd .* cc) / sum(cc .^ 2)) .* cc
    end
    return out
end
y_neutral = neutralise(y, beta)

safe_cor(a, b) = std(a) > 0 ? cor(a, b) : 0.0

function evaluate(m, dates)
    p = Float64.(m(x))
    corr = mean(safe_cor(p[rows(d)], y[rows(d)]) for d in dates)
    corr_neutral = mean(safe_cor(p[rows(d)], y_neutral[rows(d)]) for d in dates)
    dep = mean(EvoTrees.dcor2(p[rows(d)], beta[rows(d)]) for d in dates)
    dep_pooled = EvoTrees.dcor2(p[span(dates)], beta[span(dates)])
    return (; corr, corr_neutral, dep, dep_pooled)
end

show_row(name, r) = @printf("%-10s %8.4f %13.4f %8.4f %11.4f\n",
    name, r.corr, r.corr_neutral, r.dep, r.dep_pooled)
```

## Three fits

All three models share one configuration and differ only in the penalty. The penalised fits
use the weight `λ0`; how to choose it is covered further down.

```julia
config(λ; within=false) = EvoTreeRegressor(
    loss=:mse,
    nrounds=200,
    eta=0.05,
    max_depth=5,
    seed=1,
    ctrl_lambda=λ,
    ctrl_within_group=within,
)
λ0 = 10.0
```

Without a penalty:

```julia
m_base = EvoTrees.fit(config(0.0); x_train, y_train, verbosity=0)
```

With the penalty pooled over the whole training sample. The control is passed as
`ctrl_train`, one value per training row. It is a role, not a feature: the model never sees it.

```julia
m_pooled = EvoTrees.fit(config(λ0); x_train, y_train, ctrl_train=beta_train, verbosity=0)
```

With the penalty within each date. `ctrl_within_group = true` computes the statistic on each
group of `group_train` separately, so only the cross-sectional dependence is penalised. The
control is centred and scaled within each date, each date contributes in proportion to its
size, and dates below 4 rows or with a constant control are skipped.

```julia
m_within = EvoTrees.fit(config(λ0; within=true);
    x_train, y_train, ctrl_train=beta_train, group_train=date_train, verbosity=0)
```

Evaluated on the test dates:

```julia
res_base = evaluate(m_base, test_dates)
res_pooled = evaluate(m_pooled, test_dates)
res_within = evaluate(m_within, test_dates)

println("model          corr  corr_neutral      dep  dep_pooled")
show_row("none", res_base)
show_row("pooled", res_pooled)
show_row("within", res_within)
```

| **Model**  | **corr** | **corr_neutral** | **dep** | **dep_pooled** |
|------------|----------|------------------|---------|----------------|
| none       | 0.3745 | 0.3734 | 0.0334 | 0.0496 |
| pooled     | 0.3591 | 0.3824 | 0.0020 | 0.0013 |
| within     | 0.3601 | 0.3809 | 0.0027 | 0.0026 |

The unpenalised model carries a clear dependence on beta within each date. Part of its `corr`
comes from that exposure, which pays on dates the market rises and costs on dates it falls,
and it dilutes `corr_neutral`. The penalised models trade some raw `corr` for a lower `dep`.

The pooled statistic mixes two things: whether the prediction ranks assets by beta within a
date, and whether it moves with the universe's beta level across dates. Only the first
matters for a per-date ranking, and the within-date form penalises only that one.

The same fit from a table names the columns instead. The control and group columns are then
left out of the features unless listed in `feature_names`:

```julia
df_train = (; f1=x_train[:, 1], f2=x_train[:, 2], f3=x_train[:, 3], f4=x_train[:, 4],
    beta=beta_train, date=date_train, y=y_train)

m_table = EvoTrees.fit(config(λ0; within=true), df_train;
    target_name="y", ctrl_name="beta", group_name="date", verbosity=0)
m_table.info[:feature_names]
```

```
[:f1, :f2, :f3, :f4]
```

## Choosing `ctrl_lambda`

There is no default weight: the right value depends on the data, and it has to be chosen on
data the test period does not overlap. Fit a small grid on the training dates and evaluate on
the validation dates:

```julia
lambdas = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
sweep = map(lambdas) do λ
    m = EvoTrees.fit(config(λ; within=true);
        x_train, y_train, ctrl_train=beta_train, group_train=date_train, verbosity=0)
    merge((; λ, m), evaluate(m, valid_dates))
end

println("lambda         corr  corr_neutral      dep  dep_pooled")
for r in sweep
    show_row(string(r.λ), r)
end
```

| **ctrl_lambda** | **corr** | **corr_neutral** | **dep** | **dep_pooled** |
|-----------------|----------|------------------|---------|----------------|
| 0               | 0.3814 | 0.3792 | 0.0315 | 0.0423 |
| 0.1             | 0.3790 | 0.3771 | 0.0292 | 0.0393 |
| 0.3             | 0.3805 | 0.3799 | 0.0251 | 0.0336 |
| 1               | 0.3757 | 0.3801 | 0.0157 | 0.0205 |
| 3               | 0.3740 | 0.3844 | 0.0065 | 0.0079 |
| 10              | 0.3676 | 0.3865 | 0.0014 | 0.0014 |
| 30              | 0.3668 | 0.3875 | -0.0001 | 0.0001 |

Choose on the quantity the model will be judged by. Here that is the beta-neutral correlation,
so the weight with the best validation `corr_neutral` is kept and scored once on the test dates:

```julia
best = argmax(r -> r.corr_neutral, sweep)
res_best = evaluate(best.m, test_dates)
println("chosen ctrl_lambda = ", best.λ)
show_row("test", res_best)
```

The chosen weight is 30, and on the test dates it scores `corr`
0.3547, `corr_neutral` 0.3806 and `dep`
0.0013.

Where the goal is a dependence budget rather than an accuracy target, keep the smallest weight
whose validation `dep` is under the budget instead. Either way, look at the whole grid rather
than its end: past some weight the penalty overshoots, and the measured dependence can start
rising again while accuracy keeps falling.

The eval metric and early stopping see the base loss only, not the penalty, so early stopping
picks the number of rounds by accuracy alone and ignores the dependence. Without an offset the
first tree is fitted before the predictions have any spread, so it carries no penalty, and at a
strong weight the best round by the base metric can be that first one. A fixed `nrounds`, tuned
together with `ctrl_lambda`, keeps the trade-off in view.

## The scale of `ctrl_lambda`

Under `:mse` the pooled penalty adds `ctrl_lambda * W * dcov2(prediction, control)` to the loss,
`W` being the total training weight, with the control centred and scaled to unit standard
deviation over the whole sample. The within-date form adds, for each date `d`,
`ctrl_lambda * w̄ * n_d * dcov2` on that date's rows, `w̄` being the mean training weight and
`n_d` the date's number of rows, with the control centred and scaled within the date. Either
way the control's units do not matter: beta in percent or in units gives the same fit up to
rounding.
The prediction is not rescaled, and `dcov2` is not scale free: each term grows in proportion to
the spread of the prediction. The squared error grows with the square of that spread, so a
target `c` times larger needs about `c` times the weight for the same effect:

```julia
m_y10 = EvoTrees.fit(config(λ0; within=true);
    x_train, y_train=10 .* y_train, ctrl_train=beta_train, group_train=date_train, verbosity=0)
m_y10_l10x = EvoTrees.fit(config(10 * λ0; within=true);
    x_train, y_train=10 .* y_train, ctrl_train=beta_train, group_train=date_train, verbosity=0)

@printf("target x1,  lambda %g:  dep %.4f\n", λ0, res_within.dep)
@printf("target x10, lambda %g:  dep %.4f\n", λ0, evaluate(m_y10, test_dates).dep)
@printf("target x10, lambda %g: dep %.4f\n", 10 * λ0, evaluate(m_y10_l10x, test_dates).dep)
```

| **Target scale** | **ctrl_lambda** | **dep** |
|------------------|-----------------|---------|
| 1                | 10 | 0.0027 |
| 10               | 10 | 0.0173 |
| 10               | 100 | 0.0028 |

A weight tuned on one target does not carry over to a target on another scale. Retune it, or
standardise the target first.

## Other losses and devices

The penalty is available on `EvoTreeRegressor` for `:mse`, `:logloss`, `:poisson`, `:gamma` and
`:tweedie`, single and multi-target (one penalty per output), and on `EvoTreeMLE` for
`:gaussian_mle`, where it acts on the location. Under the losses other than `:mse` each row's
penalty gradient is weighted by its curvature relative to `:mse`, so a given weight has roughly
the effect it has under `:mse`. Other losses are rejected.

On GPU the base gradients are computed on the device as usual. The penalty is a sort and a sweep
over all predictions, which does not map onto a per-row kernel, so each round the penalised
gradient rows are copied to the host, the penalty is added there, and the rows are copied back.
Its cost therefore grows with the number of rows, not with the tree depth.

The MLJ interface has no way to pass a control variable, so the penalty is only available through
`EvoTrees.fit`.
