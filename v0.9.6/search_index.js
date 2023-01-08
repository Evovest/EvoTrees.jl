var documenterSearchIndex = {"docs":
[{"location":"examples/#Regression","page":"Examples","title":"Regression","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Minimal example to fit a noisy sinus wave.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: )","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using EvoTrees\nusing EvoTrees: sigmoid, logit\n\n# prepare a dataset\nfeatures = rand(10000) .* 20 .- 10\nX = reshape(features, (size(features)[1], 1))\nY = sin.(features) .* 0.5 .+ 0.5\nY = logit(Y) + randn(size(Y))\nY = sigmoid(Y)\n𝑖 = collect(1:size(X, 1))\n\n# train-eval split\n𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)\ntrain_size = 0.8\n𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]\n𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]\n\nX_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]\nY_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]\n\nparams1 = EvoTreeRegressor(\n    loss=:linear, metric=:mse,\n    nrounds=100, nbins = 100,\n    λ = 0.5, γ=0.1, η=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_linear = predict(model, X_eval)\n\n# logistic / cross-entropy\nparams1 = EvoTreeRegressor(\n    loss=:logistic, metric = :logloss,\n    nrounds=100, nbins = 100,\n    λ = 0.5, γ=0.1, η=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_logistic = predict(model, X_eval)\n\n# Poisson\nparams1 = EvoTreeCount(\n    loss=:poisson, metric = :poisson,\n    nrounds=100, nbins = 100,\n    λ = 0.5, γ=0.1, η=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\n@time pred_eval_poisson = predict(model, X_eval)\n\n# L1\nparams1 = EvoTreeRegressor(\n    loss=:L1, α=0.5, metric = :mae,\n    nrounds=100, nbins=100,\n    λ = 0.5, γ=0.0, η=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_eval_L1 = predict(model, X_eval)","category":"page"},{"location":"examples/#Quantile-Regression","page":"Examples","title":"Quantile Regression","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: )","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"# q50\nparams1 = EvoTreeRegressor(\n    loss=:quantile, α=0.5, metric = :quantile,\n    nrounds=200, nbins = 100,\n    λ = 0.1, γ=0.0, η=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q50 = predict(model, X_train)\n\n# q20\nparams1 = EvoTreeRegressor(\n    loss=:quantile, α=0.2, metric = :quantile,\n    nrounds=200, nbins = 100,\n    λ = 0.1, γ=0.0, η=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q20 = predict(model, X_train)\n\n# q80\nparams1 = EvoTreeRegressor(\n    loss=:quantile, α=0.8,\n    nrounds=200, nbins = 100,\n    λ = 0.1, γ=0.0, η=0.05,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0)\nmodel = fit_evotree(params1, X_train, Y_train, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 25)\npred_train_q80 = predict(model, X_train)","category":"page"},{"location":"examples/#Gaussian-Max-Likelihood","page":"Examples","title":"Gaussian Max Likelihood","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"(Image: )","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"params1 = EvoTreeGaussian(\n    loss=:gaussian, metric=:gaussian,\n    nrounds=100, nbins=100,\n    λ = 0.0, γ=0.0, η=0.1,\n    max_depth = 6, min_weight = 1.0,\n    rowsample=0.5, colsample=1.0, seed=123)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Docs under development, see README in the meantime.","category":"page"},{"location":"","page":"Home","title":"Home","text":"fit_evotree\npredict\nimportance","category":"page"},{"location":"#EvoTrees.fit_evotree","page":"Home","title":"EvoTrees.fit_evotree","text":"fit_evotree(params, X_train, Y_train, W_train=nothing;\n    X_eval=nothing, Y_eval=nothing, W_eval = nothing,\n    early_stopping_rounds=9999,\n    print_every_n=9999,\n    verbosity=1)\n\nMain training function. Performorms model fitting given configuration params, X_train, Y_train input data. \n\nArguments\n\nparams::EvoTypes: configuration info providing hyper-paramters. EvoTypes comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount or EvoTreeGaussian\nX_train::Matrix: training data of size [#observations, #features]. \nY_train::Vector: vector of train targets of length #observations.\nW_train::Vector: vector of train weights of length #observations. Defaults to nothing and a vector of ones is assumed.\n\nKeyword arguments\n\nX_eval::Matrix: evaluation data of size [#observations, #features]. \nY_eval::Vector: vector of evaluation targets of length #observations.\nW_eval::Vector: vector of evaluation weights of length #observations. Defaults to nothing (assumes a vector of 1s).\nearly_stopping_rounds::Integer: number of consecutive rounds without metric improvement after which fitting in stopped. \nprint_every_n: sets at which frequency logging info should be printed. \nverbosity: set to 1 to print logging info during training.\n\n\n\n\n\n","category":"function"},{"location":"#MLJModelInterface.predict","page":"Home","title":"MLJModelInterface.predict","text":"predict(loss::L, tree::Tree{T}, X::AbstractMatrix, K)\n\nPrediction from a single tree - assign each observation to its final leaf.\n\n\n\n\n\npredict(model::GBTree{T}, X::AbstractMatrix)\n\nPredictions from an EvoTrees model - sums the predictions from all trees composing the model.\n\n\n\n\n\n","category":"function"},{"location":"#EvoTrees.importance","page":"Home","title":"EvoTrees.importance","text":"importance(model::GBTree, vars::AbstractVector)\n\nSorted normalized feature importance based on loss function gain.\n\n\n\n\n\n","category":"function"}]
}