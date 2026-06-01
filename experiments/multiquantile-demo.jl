using Random
using Statistics
using EvoTrees
# using CUDA

Random.seed!(123)

n = 2000
x = rand(n)
sigma = 0.1 .+ 0.5 .* x
y = sin.(2pi .* x) .+ sigma .* randn(n)
X = reshape(x, :, 1)

idx = randperm(n)
n_train = Int(round(0.8 * n))
train_idx = idx[1:n_train]
eval_idx = idx[n_train+1:end]

x_train, x_eval = X[train_idx, :], X[eval_idx, :]
y_train, y_eval = y[train_idx], y[eval_idx]

alphas = [0.2, 0.5, 0.9]
config = EvoTreeRegressor(
    loss=:multiquantile,
    alphas=alphas,
    nrounds=100,
    nbins=64,
    lambda=0.0,
    gamma=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.8,
    colsample=1.0,
    seed=123,
    device=:gpu,
)

model = EvoTrees.fit(
    config;
    x_train,
    y_train,
    x_eval=x_eval,
    y_eval=y_eval,
    print_every_n=50,
)

preds = EvoTrees.predict(model, x_eval)
coverage = [mean(y_eval .<= preds[:, k]) for k in eachindex(alphas)]
println("alphas = ", alphas)
println("coverage= ", round.(coverage; digits=3))

K = length(alphas)
crossings = [mean(preds[:, k] .> preds[:, k + 1]) for k in 1:K-1]
any_cross = mean(vec(any(diff(preds, dims=2) .< 0, dims=2)))
println("crossing rates (q_k > q_{k+1}) = ", round.(crossings; digits=4))
println("any crossing rate = ", round(any_cross; digits=4))

calib = hcat(alphas, coverage, coverage .- alphas)
show(stdout, "text/plain", calib)
println()

order = sortperm(vec(x_eval))
x_sorted = vec(x_eval)[order]
preds_sorted = preds[order, :]
preview = hcat(x_sorted[1:10], preds_sorted[1:10, :])
show(stdout, "text/plain", preview)
println()
