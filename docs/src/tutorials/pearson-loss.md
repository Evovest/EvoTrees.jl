# Per-date correlation

On a panel of stock returns by date and asset, a model is often scored by the Pearson
correlation between prediction and target within each date, averaged over dates.
`loss = :pearson` trains for that score, with one group per date.

```julia
using EvoTrees
using Random
using Statistics
using Printf
```

## A synthetic panel

Each date holds 200 to 400 assets and has its own volatility and level. The level is partly
predictable from `z`, a feature constant within a date. Within a date the target carries a weak
signal in `x1` to `x4` plus heavy-tailed noise.

```julia
rng = Xoshiro(123)
ndates = 400
sizes = rand(rng, 200:400, ndates)
date = reduce(vcat, [fill(d, n) for (d, n) in enumerate(sizes)])
nobs = length(date)
rows = [(s - n + 1):s for (s, n) in zip(cumsum(sizes), sizes)]

vol = 0.02 .* exp.(0.5 .* randn(rng, ndates))
z = randn(rng, ndates)
level = 3 .* vol .* (0.8 .* z .+ 0.6 .* randn(rng, ndates))

x1, x2, x3, x4, x5 = (randn(rng, nobs) for _ in 1:5)
f = x1 .+ 0.7 .* sin.(2 .* x2) .+ 0.5 .* x3 .* x4
signal = similar(f)
for r in rows
    signal[r] = (f[r] .- mean(f[r])) ./ std(f[r]; corrected=false)
end

chi2 = randn(rng, nobs) .^ 2 .+ randn(rng, nobs) .^ 2 .+ randn(rng, nobs) .^ 2
noise = randn(rng, nobs) ./ sqrt.(chi2 ./ 3) ./ sqrt(3)          # Student-t(3), variance 1

rho = 0.1
y = level[date] .+ vol[date] .* (rho .* signal .+ sqrt(1 - rho^2) .* noise)
x = hcat(x1, x2, x3, x4, x5, z[date])

span(ds) = first(rows[first(ds)]):last(rows[last(ds)])
train_dates, valid_dates, test_dates = 1:240, 241:320, 321:400
train, valid, test = span(train_dates), span(valid_dates), span(test_dates)

safe_cor(a, b) = std(a) > 0 ? cor(a, b) : 0.0
date_cor(p, dates) = mean(safe_cor(p[rows[d]], y[rows[d]]) for d in dates)
```

## Fitting

Groups are passed through `group_train` and `group_eval` (or `group_name` when fitting from a
table). Both fits stop early on `metric = :pearson`.

```julia
function fit_and_score(loss)
    config = EvoTreeRegressor(loss=loss, metric=:pearson, nrounds=3000,
        early_stopping_rounds=100, eta=0.05, max_depth=4, seed=1)
    m = EvoTrees.fit(config;
        x_train=x[train, :], y_train=y[train], group_train=date[train],
        x_eval=x[valid, :], y_eval=y[valid], group_eval=date[valid],
        verbosity=0)
    best_iter = m.info[:logger][:best_iter]
    p = zeros(nobs)
    p[test] = m(x[test, :]; ntree_limit=best_iter)
    return (; loss, best_iter, test_r=date_cor(p, test_dates))
end

for r in (fit_and_score(:mse), fit_and_score(:pearson))
    @printf("%-10s %6d %8.4f\n", r.loss, r.best_iter, r.test_r)
end
```

| **Loss**   | **Best iteration** | **Test per-date r** |
|------------|--------------------|---------------------|
| `:mse`     | 125                | 0.0783              |
| `:pearson` | 136                | 0.0994              |

The true signal, `date_cor(signal, test_dates)`, scores 0.1123. `:mse` on the raw target is
dominated by the most volatile dates and can spend splits on `z`, which cannot help within a date.

## Notes

- Each date's prediction, centred on its own mean, is fitted to the date's standardised target
  by squared error. Within a date this error is `(s - r)^2 + 1 - r^2` for prediction spread `s`
  and correlation `r`.
- For a Spearman-style variant, replace the target by its ranks (or their normal quantiles)
  within each date before fitting.
- Predictions are scores within a date: their level carries no meaning across dates.
- A date weighs by its number of rows. To count dates equally, pass
  `w_train = mean(sizes[train_dates]) ./ sizes[date[train]]`.
- An offset is part of the centred prediction, so give it on the scale of a `:pearson` model's
  output, such as a previous `:pearson` model's predictions.
