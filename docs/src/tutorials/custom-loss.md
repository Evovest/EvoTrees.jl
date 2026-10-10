# Custom loss

`loss = :custom` fits an arbitrary twice differentiable loss. You pass the loss as a function of
one prediction and its target, and the gradient and hessian that gradient boosting needs are
taken by automatic differentiation through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl).

DifferentiationInterface is a package extension, so it is not installed with EvoTrees. Load it
along with a backend, and pass the backend as `loss_backend`:

```julia
using EvoTrees, DifferentiationInterface, ForwardDiff

config = EvoTreeRegressor(
    loss = :custom,
    loss_fn = (p, y) -> (p - y)^2,
    loss_backend = AutoForwardDiff(),
    nrounds = 100,
)
m = EvoTrees.fit(config; x_train, y_train)
```

`loss_fn(p, y)` takes a single raw prediction and its target and returns a scalar. It is
differentiated with respect to `p`, so the target may be of any type the function accepts.

## Predictions are raw scores

The built-in losses apply an inverse link in `predict`: `:logloss` returns a probability,
`:poisson` returns a rate. A custom loss has no link EvoTrees knows about, so `predict` returns
the raw score and mapping it back is yours to do.

```julia
using EvoTrees: sigmoid

logloss = (p, y) -> log(1 + exp(p)) - y * p
m = EvoTrees.fit(EvoTreeRegressor(loss = :custom, loss_fn = logloss,
                                  loss_backend = AutoForwardDiff(), nrounds = 100);
                 x_train, y_train)
probs = sigmoid.(EvoTrees.predict(m, x_train))
```

## The initial prediction

Training starts from the constant that minimises the loss over the training target, found by a
Newton solve. For a squared loss that is the mean of the target, and for the logistic loss above
it is `logit(mean(y))`, which is where the built-in `:mse` and `:logloss` also start. When the
loss has no positive curvature at the solve's iterate, the initial prediction is 0.

## Why you would use one

A Huber loss is robust to outliers and EvoTrees has no built-in equivalent. On 4000 rows with
5 features where 8% of the training targets are gross outliers, scored against the clean signal
on 2000 held out rows, 200 rounds at `eta = 0.1`:

| model | test MAE |
|---|---|
| built-in `:mse` | 3.331 |
| built-in `:mae` | 3.373 |
| custom Huber, `delta = 1`, `lambda = 0` | 0.207 |
| custom Huber, `delta = 1`, `lambda = 2` | 0.071 |

```julia
huber = (p, y) -> (r = p - y; abs(r) <= 1 ? r^2 / 2 : abs(r) - 0.5)
config = EvoTreeRegressor(loss = :custom, loss_fn = huber, loss_backend = AutoForwardDiff(),
                          lambda = 2.0, nrounds = 200, eta = 0.1)
```

## Losses with a flat region

Gradient boosting sizes a leaf as `-eta * sum(g) / (sum(h) + lambda * sum(w) + L2)`. A loss whose
second derivative is zero over part of its range, a Huber loss outside its band for instance,
contributes nothing to `sum(h)` there, and the leaf value then grows with the number of
observations in the node instead of staying scale free. That is what the `lambda` column above is
doing, and with a larger `eta` it is the difference between converging and not.

The same Huber loss on clean data at `eta = 0.3`, where most residuals leave the band early:

| `lambda` | test MAE |
|---|---|
| 0.0 | 8.241 |
| 0.1 | 0.108 |
| 1.0 | 0.051 |

`lambda` multiplies the node weight in that denominator, which is exactly what the built-in
`:mae` leaf rule uses in place of the hessian sum, so it is the knob for this case. The built-in
`:mae` reaches 0.056 on that same clean data.

## Limitations

- Single target only. A matrix target is rejected.
- CPU only. The GPU kernels call the built-in loss expressions directly.
- The model holds a reference to `loss_fn`, so a saved model needs that function in scope to be
  loaded again.
