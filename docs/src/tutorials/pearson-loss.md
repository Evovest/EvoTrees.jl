# Pearson loss

`loss = :pearson` fits the Pearson correlation between the predictions and the target within each
group. The roles of the inputs are:

- `X`: the input features;
- `Y`: the target;
- `group`: the group of each row.

Each group's centred prediction is fitted to its standardised target by squared error, as described
in the [design notes](../pearson-design.md).

## Data

Each group has its own level and scale. The last feature is constant within a group and partly
predicts its level. Within a group, `Y` carries a weak signal from the first four features and
heavy-tailed noise.

```julia
using EvoTrees, Random, Statistics, Printf

function simulate(rng; ngroups)
    sizes = rand(rng, 200:400, ngroups)
    group = reduce(vcat, [fill(g, n) for (g, n) in enumerate(sizes)])
    n = length(group)
    scale = exp.(0.5 .* randn(rng, ngroups))
    z = randn(rng, ngroups)
    level = 3 .* scale .* (0.8 .* z .+ 0.6 .* randn(rng, ngroups))
    x = hcat(randn(rng, n, 5), z[group])
    signal = x[:, 1] .+ 0.7 .* sin.(2 .* x[:, 2]) .+ 0.5 .* x[:, 3] .* x[:, 4]
    noise = randn(rng, n) ./ sqrt.(randn(rng, n) .^ 2 .+ randn(rng, n) .^ 2 .+ randn(rng, n) .^ 2)
    y = level[group] .+ scale[group] .* (0.1 .* signal .+ noise)
    return (; x, y, group)
end

rng = Xoshiro(123)
train, valid, test = simulate(rng; ngroups=240), simulate(rng; ngroups=80), simulate(rng; ngroups=80)
```

## Fitting

The groups are passed as `group_train` and `group_eval`. From a table, pass the name of the group
column as `group_name`. Both models stop early on `metric = :pearson`, the mean correlation within
the evaluation groups.

```julia
config(loss) = EvoTreeRegressor(loss=loss, metric=:pearson, nrounds=3000,
    early_stopping_rounds=100, eta=0.05, max_depth=4)

m_mse = EvoTrees.fit(config(:mse); x_train=train.x, y_train=train.y, group_train=train.group,
    x_eval=valid.x, y_eval=valid.y, group_eval=valid.group, verbosity=0)
m_pearson = EvoTrees.fit(config(:pearson); x_train=train.x, y_train=train.y, group_train=train.group,
    x_eval=valid.x, y_eval=valid.y, group_eval=valid.group, verbosity=0)
```

On the test set, compare the mean correlation within groups at each model's best round:

```julia
rows = [findall(==(g), test.group) for g in unique(test.group)]
for (name, m) in ((":mse", m_mse), (":pearson", m_pearson))
    best = m.info[:logger][:best_iter]
    p = m(test.x; ntree_limit=best)
    @printf("%-9s  best round %4d  mean cor(p, Y) within groups %.4f\n", name, best,
        mean(cor(p[r], test.y[r]) for r in rows))
end
```

```
:mse       best round  164  mean cor(p, Y) within groups 0.0895
:pearson   best round   88  mean cor(p, Y) within groups 0.1117
```

`:mse` also fits the level of each group, mostly through the last feature, and its error is
dominated by the groups with the largest scale. `:pearson` only fits the variation within each
group, with every group on the same scale.

## Notes

Groups with fewer than two rows or a constant target are ignored. Predictions are only comparable
within a group.

Each row counts once, so a group weighs by its total weight. To count groups equally, give each
row a weight inversely proportional to the size of its group. An offset is part of the centred
prediction, so give it on the scale of a `:pearson` model's output.

For a Spearman-style correlation, replace the target by its ranks within each group before
fitting.
