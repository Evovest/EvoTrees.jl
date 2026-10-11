# Decorrelation penalty

`ctrl_lambda` penalises the dependence between the prediction and a control variable, such as an
asset's market beta in a stock return model, so the trees favour the part of the signal the control
does not explain. The dependence is the unbiased squared distance covariance `dcov2`, which is zero
in population only under independence, so nonlinear dependence is penalised too.

This tutorial fits a synthetic panel of assets over dates without a penalty, with the penalty over
the whole sample, and within each date, then shows how to choose `ctrl_lambda`.

## Getting started

```julia
using EvoTrees
using Random
using Statistics
using Printf
```

## A synthetic panel

Each row is an asset on a date. The control is the asset's beta: a level shared by all assets on a
date plus the asset's own part. The return is beta times the market's return on that date, plus
`alpha`, the part a model should find, plus noise. The market's return is positive on average, so a
model is rewarded for favouring high beta assets.

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

Dates are split in order into train, validation and test periods:

```julia
rows(d) = ((d - 1) * nassets + 1):(d * nassets)
span(dates) = first(rows(first(dates))):last(rows(last(dates)))

train_dates, valid_dates, test_dates = 1:250, 251:325, 326:400
train = span(train_dates)

x_train, y_train = x[train, :], y[train]
beta_train, date_train = beta[train], date[train]
```

## Measuring dependence per date

Each quantity below is computed within each date and averaged over dates:

- `corr`: Pearson correlation of the prediction with the target;
- `corr_neutral`: the same against the target's residual after regressing it on beta within the date;
- `dep`: bias-corrected squared distance correlation of the prediction with beta, near zero (and
  possibly slightly negative) under independence;
- `dep_pooled`: the same over all rows of the period at once.

`EvoTrees.dcor2` (internal, not exported) computes the squared distance correlation.

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

The three models differ only in the penalty. Choosing `λ0` is covered further down.

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

With the penalty over the whole training sample. The control is passed as `ctrl_train`, one value
per training row, and is not a feature.

```julia
m_pooled = EvoTrees.fit(config(λ0); x_train, y_train, ctrl_train=beta_train, verbosity=0)
```

With the penalty within each date. `ctrl_within_group = true` computes the statistic on each group
of `group_train` separately, with the control centred and scaled within the group.

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

The unpenalised model depends on beta within each date. The penalised models trade some `corr` for
a lower `dep` and a higher `corr_neutral`.

Under `:mse` the within-date penalty does not change when a constant is added to a date's
predictions, so it puts no net push on any date's mean prediction: it removes the dependence within
dates and leaves differences between dates to the loss.

From a table, name the columns instead. The control and group columns are not used as features
unless listed in `feature_names`:

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

The right weight depends on the data. Fit a grid on the training dates and evaluate on the
validation dates:

```julia
lambdas = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
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
| 100             | 0.3376 | 0.3554 | -0.0006 | 0.0001 |
| 300             | 0.2757 | 0.2767 | 0.0084 | 0.0131 |

Keep the weight with the best validation score on the quantity the model is judged by, here
`corr_neutral`, and score it once on the test dates:

```julia
best = argmax(r -> r.corr_neutral, sweep)
res_best = evaluate(best.m, test_dates)
println("chosen ctrl_lambda = ", best.λ)
show_row("test", res_best)
```

The chosen weight is 30, and on the test dates it scores `corr`
0.3547, `corr_neutral` 0.3806 and `dep`
0.0013.

For a dependence budget instead, keep the smallest weight whose validation `dep` is under it. Past
some weight the penalty overshoots: here `corr_neutral` falls from 100 on, and at 300 `dep` rises
again. Before keeping a weight at the edge of a grid, extend the grid past it.

### Early stopping

With evaluation data, the loss's own metric becomes the penalised objective on the evaluation set,
and early stopping follows it. The evaluation set needs its own controls and, within dates, its own
dates:

```julia
valid = span(valid_dates)
es_config = EvoTreeRegressor(loss=:mse, nrounds=2000, eta=0.05, max_depth=5, seed=1,
    ctrl_lambda=λ0, ctrl_within_group=true, early_stopping_rounds=50)
m_es = EvoTrees.fit(es_config; x_train, y_train, ctrl_train=beta_train, group_train=date_train,
    x_eval=x[valid, :], y_eval=y[valid], ctrl_eval=beta[valid], group_eval=date[valid], verbosity=0)

lg = m_es.info[:logger]
b = lg[:best_iter] + 1
@printf("best round %d: mse %.4f, dependence %.5f, penalised %.4f\n",
    lg[:best_iter], lg[:base_metrics][b], lg[:ctrl_dependence][b][1], lg[:metrics][b])
```

| **best round** | **mse** | **dependence** | **penalised** |
|----------------|---------|----------------|---------------|
| 51 | 2.5243 | 0.00022 | 2.5265 |

The penalised value is the mse plus `ctrl_lambda` times the dependence (here the size-weighted mean
of each date's `dcov2`). The logger keeps both parts for every round. Other metrics, such as
`:pearson`, are reported unchanged. Predict with `ntree_limit=lg[:best_iter]` to use the round early
stopping chose. Without an offset the first tree is fitted before the penalty acts, so at a strong
weight the best round can be round 0.

## The scale of `ctrl_lambda`

Under `:mse` the penalty adds `ctrl_lambda * W * dcov2(prediction, control)` to the loss, `W` being
the total training weight. Within dates it adds `ctrl_lambda * w̄ * n_d * dcov2` for each date `d`,
`w̄` being the mean training weight and `n_d` the date's number of rows. The control is standardised,
so its units do not matter. The prediction is not: `dcov2` grows linearly with its spread and the
squared error quadratically, so a target `c` times larger needs about `c` times the weight:

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

Retune the weight for a target on another scale, or standardise the target first.

## Several controls

`ctrl_train` also takes a matrix with one column per control, and `ctrl_name` a vector of names.
Each control adds its own term, multiplied by its entry in `ctrl_weights` (1 by default). Here the
volatility feature, a noisy proxy for beta, is added at half the weight of beta:

```julia
vol = x[:, 2]
m_two = EvoTrees.fit(config(λ0; within=true); x_train, y_train,
    ctrl_train=hcat(beta_train, vol[train]), ctrl_weights=[1.0, 0.5],
    group_train=date_train, verbosity=0)

function dep_on(m, c, dates)
    p = Float64.(m(x))
    return mean(EvoTrees.dcor2(p[rows(d)], c[rows(d)]) for d in dates)
end
for (name, m) in (("none", m_base), ("beta", m_within), ("beta, vol", m_two))
    @printf("%-10s dep on beta %.4f  dep on vol %.4f\n", name,
        dep_on(m, beta, test_dates), dep_on(m, vol, test_dates))
end
```

| **Controls** | **dep on beta** | **dep on vol** |
|--------------|-----------------|----------------|
| none         | 0.0334 | 0.0302 |
| beta         | 0.0027 | 0.0005 |
| beta, vol    | 0.0024 | 0.0001 |

Penalising beta alone already removes most of the dependence on volatility, and the second control
removes more. The terms are added, so a dependence that shows only in a combination of controls is
not penalised. Each control adds a sort and a sweep per round.

## Other losses and devices

The penalty is available on `EvoTreeRegressor` for `:mse`, `:logloss`, `:poisson`, `:gamma` and
`:tweedie`, and on `EvoTreeMLE` for `:gaussian_mle`, where it acts on the location. Under losses
other than `:mse` the penalty gradient is scaled by the loss curvature, so a given weight has about
the same effect.

On GPU the penalised gradient rows are copied to the host each round, penalised there and copied
back, so the cost grows with the number of rows.

The MLJ interface cannot pass a control variable, so the penalty is only available through
`EvoTrees.fit`.
