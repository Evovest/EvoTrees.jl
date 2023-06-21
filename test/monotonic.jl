@testset "Monotonic Constraints" begin

    using Statistics
    using StatsBase: sample
    using EvoTrees
    using EvoTrees: sigmoid, logit

    # prepare a dataset
    features = rand(10_000) .* 2.5
    X = reshape(features, (size(features)[1], 1))
    Y = sin.(features) .* 0.5 .+ 0.5
    Y = logit(Y) + randn(size(Y)) .* 0.2
    Y = sigmoid(Y)
    is = collect(1:size(X, 1))
    seed = 123

    # train-eval split
    i_sample = sample(is, size(is, 1), replace=false)
    train_size = 0.8
    i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
    i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

    x_train, x_eval = X[i_train, :], X[i_eval, :]
    y_train, y_eval = Y[i_train], Y[i_eval]

    ######################################
    ### MSE - CPU
    ######################################
    # benchmark
    params1 = EvoTreeRegressor(
        device="cpu",
        loss=:mse,
        nrounds=200,
        nbins=32,
        lambda=1.0,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=0.0,
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:mse, print_every_n=25)
    preds_ref = EvoTrees.predict(model, x_train);

    # monotonic constraint
    params1 = EvoTreeRegressor(
        device="cpu",
        loss=:mse,
        nrounds=200,
        nbins=32,
        lambda=1.0,
        gamma=0.0,
        eta=0.5,
        max_depth=6,
        min_weight=0.0,
        monotone_constraints=Dict(1 => 1),
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:mse, print_every_n=25)
    preds_mono = EvoTrees.predict(model, x_train);

    # using Plots
    # x_perm = sortperm(x_train[:, 1])
    # plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
    # plot!(x_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
    # plot!(x_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


    ######################################
    ### MSE - GPU
    ######################################
    # benchmark
    # params1 = EvoTreeRegressor(
    #     device="gpu",
    #     loss=:mse,
    #     nrounds=200, nbins=32,
    #     lambda=1.0, gamma=0.0, eta=0.05,
    #     max_depth=6, min_weight=0.0,
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:mse, print_every_n=25);
    # preds_ref = predict(model, x_train);

    # # monotonic constraint
    # params1 = EvoTreeRegressor(
    #     device="gpu",
    #     loss=:mse,
    #     nrounds=200, nbins=32,
    #     lambda=1.0, gamma=0.0, eta=0.5,
    #     max_depth=6, min_weight=0.0,
    #     monotone_constraints=Dict(1 => 1),
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:mse, print_every_n=25);
    # preds_mono = predict(model, x_train);

    # using Plots
    # x_perm = sortperm(x_train[:, 1])
    # plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
    # plot!(x_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference - GPU")
    # plot!(x_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic - GPU")


    ######################################
    ### Logloss - CPU
    ######################################
    # benchmark
    params1 = EvoTreeRegressor(
        device="cpu",
        loss=:logloss,
        nrounds=200,
        nbins=32,
        lambda=0.05,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=0.0,
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:logloss, print_every_n=25)
    preds_ref = predict(model, x_train)

    # monotonic constraint
    params1 = EvoTreeRegressor(
        device="cpu",
        loss=:logloss,
        nrounds=200,
        nbins=32,
        lambda=0.05,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=0.0,
        monotone_constraints=Dict(1 => 1),
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:logloss, print_every_n=25)
    preds_mono = predict(model, x_train)

    # x_perm = sortperm(x_train[:, 1])
    # plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
    # plot!(x_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
    # plot!(x_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


    ######################################
    ### LogLoss - GPU
    ######################################
    # benchmark
    # params1 = EvoTreeRegressor(
    #     device="gpu",
    #     loss=:logloss, metric=:logloss,
    #     nrounds=200, nbins=32,
    #     lambda=0.05, gamma=0.0, eta=0.05,
    #     max_depth=6, min_weight=0.0,
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
    # preds_ref = EvoTrees.predict(model, x_train);

    # # monotonic constraint
    # params1 = EvoTreeRegressor(
    #     device="gpu",
    #     loss=:logloss, metric=:logloss,
    #     nrounds=200, nbins=32,
    #     lambda=0.05, gamma=0.0, eta=0.05,
    #     max_depth=6, min_weight=0.0,
    #     monotone_constraints=Dict(1 => 1),
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
    # preds_mono = EvoTrees.predict(model, x_train);

    # using Plots
    # using Colors
    # x_perm = sortperm(X_train[:, 1])
    # plot(X_train, Y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
    # plot!(X_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
    # plot!(X_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


    ######################################
    ### Gaussian - CPU
    ######################################
    # benchmark
    params1 = EvoTreeGaussian(
        device="cpu",
        metric=:gaussian,
        nrounds=200,
        nbins=32,
        lambda=1.0,
        gamma=0.0,
        eta=0.05,
        max_depth=6,
        min_weight=0.0,
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:gaussian_mle, print_every_n=25)
    preds_ref = predict(model, x_train)

    # monotonic constraint
    params1 = EvoTreeGaussian(
        device="cpu",
        metric=:gaussian,
        nrounds=200,
        nbins=32,
        lambda=1.0,
        gamma=0.0,
        eta=0.5,
        max_depth=6,
        min_weight=0.0,
        monotone_constraints=Dict(1 => 1),
        rowsample=0.5,
        colsample=1.0,
        rng=seed,
    )

    model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, metric=:gaussian_mle, print_every_n=25)
    preds_mono = EvoTrees.predict(model, x_train)

    # x_perm = sortperm(x_train[:, 1])
    # plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="")
    # plot!(x_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
    # plot!(x_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")


    ######################################
    ### Gaussian - GPU
    ######################################
    # benchmark
    # params1 = EvoTreeGaussian(
    #     device="gpu",
    #     metric=:gaussian,
    #     nrounds=200, nbins=32,
    #     lambda=1.0, gamma=0.0, eta=0.05,
    #     max_depth=6, min_weight=0.0,
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
    # preds_ref = EvoTrees.predict(model, x_train)

    # # monotonic constraint
    # params1 = EvoTreeGaussian(
    #     device="gpu",
    #     metric=:gaussian,
    #     nrounds=200, nbins=32,
    #     lambda=1.0, gamma=0.0, eta=0.5,
    #     max_depth=6, min_weight=0.0,
    #     monotone_constraints=Dict(1 => 1),
    #     rowsample=0.5, colsample=1.0, rng=seed)

    # model = fit_evotree(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
    # preds_mono = EvoTrees.predict(model, x_train)

    # x_perm = sortperm(x_train[:, 1])
    # plot(x_train, y_train, msize=1, mcolor="gray", mswidth=0, background_color=RGB(1, 1, 1), seriestype=:scatter, xaxis=("feature"), yaxis=("target"), legend=true, label="GPU Gauss")
    # plot!(x_train[:, 1][x_perm], preds_ref[x_perm], color="navy", linewidth=1.5, label="Reference")
    # plot!(x_train[:, 1][x_perm], preds_mono[x_perm], color="red", linewidth=1.5, label="Monotonic")

end