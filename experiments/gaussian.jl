using Plots
using Statistics
using StatsBase: sample
using Distributions
using Revise
using EvoTrees

features = rand(Int(1.25e4), 5)
# prepare a dataset
# features = rand(100, 10)
X = features
Y = randn(size(X, 1)) .* 0.1
Y[X[:,1] .< 0.2] .*= 2
Y[(X[:,1] .>= 0.4) .& (X[:,1] .< 0.6)] .*= 5
Y[(X[:,1] .>= 0.9)] .*= 5
𝑖 = collect(1:size(X,1))

Y .*= 0.01

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# train model
params1 = EvoTreeGaussian(
    loss=:gaussian, metric=:gaussian,
    nrounds=40,
    λ = 0.0, γ=0.0, η=0.05,
    max_depth = 5, min_weight = 50.0,
    rowsample=0.5, colsample=1.0, nbins=200)

@time model = fit_evotree(params1, X_train, Y_train, X_eval=X_eval, Y_eval=Y_eval, print_every_n = 10);
# @time model = fit_evotree(params1, X_train, Y_train, print_every_n = 10);
@time pred_train = EvoTrees.predict(model, X_train)
@time pred_train_gauss = EvoTrees.predict(params1, model, X_train)

pred_gauss = [Distributions.Normal(pred_train[i,1], pred_train[i,2]) for i in 1:size(pred_train,1)]
pred_q90 = quantile.(pred_gauss, 0.9)
pred_q10 = quantile.(pred_gauss, 0.1)

mean(Y_train .< pred_q90)
mean(Y_train .< pred_q10)

x_perm = sortperm(X_train[:,1])
plot(X_train[:, 1], Y_train, ms = 1, mcolor = "gray", mscolor = "lightgray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train[x_perm, 1], color = "navy", linewidth = 1.5, label = "mu")
plot!(X_train[:,1][x_perm], pred_train[x_perm, 2], color = "blue", linewidth = 1.5, label = "sigma")
plot!(X_train[:,1][x_perm], pred_q10[x_perm, 1], color = "red", linewidth = 1.5, label = "q10")
plot!(X_train[:,1][x_perm], pred_q90[x_perm, 1], color = "green", linewidth = 1.5, label = "q90")
savefig("regression_gaussian_v1.png")


# compare with zygote
using Zygote

pred = [0.0, log(1.0)]
target = 0.1

δ1 = (target - pred[1]) / max(1e-8, exp(2*pred[2]))
δ2 = (1 - (pred[1] - target)^2 / max(1e-8, exp(2*pred[2])))

δ²1 = 1 / max(1e-8, exp(2*pred[2]))
δ²2 = 2 / max(1e-8, exp(2*pred[2])) * (pred[1] - target)^2


lpdf(x,μ,σ) = -log(σ) - log(2π)/2 - 1/2*((x-μ)/σ)^2
lpdf(0, pred[1], pred[2])

lpdf2(x,μ,lσ) = -log(exp(lσ)) - log(2π)/2 - 1/2*((x-μ)/exp(lσ))^2
lpdf2(0, pred[1], pred[2])


n1 = Normal(0, 1)
Distributions.logpdf(n1, 0)

# gradient(lpdf, target, pred[1], pred[2])[2:end]
gradient(lpdf2, target, pred[1], pred[2])[2:end]
Zygote.hessian(lpdf2, target, pred[1], pred[2])

gradient_lpdf(x,pred) = gradient(lpdf2, x, pred[1], pred[2])[3]
hessian_lpdf(x,pred) = gradient(gradient_lpdf, x, pred)[1]
gradient_lpdf(target, pred)
hessian_lpdf(target, pred)
