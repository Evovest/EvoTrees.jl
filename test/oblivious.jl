using Statistics
using StatsBase: sample
using EvoTrees: sigmoid, logit
using EvoTrees: check_args, check_parameter
using Random: seed!

# prepare a dataset
seed!(123)
features = rand(1_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size*size(is, 1))]
i_eval = i_sample[(floor(Int, train_size*size(is, 1))+1):end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

Yc = (Y .> 0.8) .+ 1
y_train_c, y_eval_c = Yc[i_train], Yc[i_eval]

@testset "oblivious regressor" begin
    @testset for loss in [:mse, :logloss, :quantile, :mae, :gamma, :tweedie]
        config = EvoTreeRegressor(
            loss=loss,
            tree_type=:oblivious,
            nrounds=100,
            nbins=32,
            seed=123,
            eta=0.1,
        )

        model, cache = EvoTrees.init(config, x_train, y_train)
        preds_ini = model(x_eval)
        mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
        model = fit(
            config;
            x_train,
            y_train,
            x_eval,
            y_eval,
            print_every_n=100
        )

        preds = model(x_eval)
        mse_error = mean(abs.(preds .- y_eval) .^ 2)
        mse_gain_pct = mse_error / mse_error_ini - 1
        @test mse_gain_pct < -0.75

    end
end

@testset "oblivious count" begin

    config = EvoTreeCount(
        tree_type="oblivious",
        nrounds=100,
        nbins=32,
        seed=123,
    )

    model, cache = EvoTrees.init(config, x_train, y_train)
    preds_ini = model(x_eval)
    mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
    model = fit(
        config;
        x_train,
        y_train,
        x_eval,
        y_eval,
        print_every_n=100
    )

    preds = model(x_eval)
    mse_error = mean(abs.(preds .- y_eval) .^ 2)
    mse_gain_pct = mse_error / mse_error_ini - 1
    @test mse_gain_pct < -0.75

end

@testset "oblivious MLE" begin
    @testset for loss in [:gaussian_mle, :logistic_mle]

        config = EvoTreeMLE(
            loss=loss,
            tree_type="oblivious",
            nrounds=100,
            nbins=32,
            seed=123,
        )

        model, cache = EvoTrees.init(config, x_train, y_train)
        preds_ini = model(x_eval)[:, 1]
        mse_error_ini = mean(abs.(preds_ini .- y_eval) .^ 2)
        model = fit(
            config;
            x_train,
            y_train,
            x_eval,
            y_eval,
            print_every_n=100
        )

        preds = model(x_eval)
        @test all(>(0), preds[:, 2])
        mse_error = mean(abs.(preds[:, 1] .- y_eval) .^ 2)
        mse_gain_pct = mse_error / mse_error_ini - 1
        @test mse_gain_pct .< 0.75

    end
end

@testset "oblivious classifier" begin

    config = EvoTreeClassifier(
        tree_type="oblivious",
        nrounds=100,
        nbins=32,
        seed=123,
    )

    model, cache = EvoTrees.init(config, x_train, y_train_c)
    preds_ini = model(x_eval)
    acc_ini = mean(map(argmax, eachrow(preds_ini)) .== y_eval_c)

    model = fit(
        config;
        x_train,
        y_train=y_train_c,
        x_eval,
        y_eval=y_eval_c,
        print_every_n=100
    )

    preds = model(x_eval)
    acc = mean(map(argmax, eachrow(preds)) .== y_eval_c)

    @test acc > 0.85

end

@testset "oblivious importance counts each depth once" begin
    # A depth of an oblivious tree shares one split, and its gain is summed over every node of
    # the depth. `importance` adds `tree.gain` once per split node, so storing the depth total on
    # each node counted depth `d` 2^(d-1) times. Here binary and oblivious fit the same tree, so
    # their importance has to agree.
    seed!(7)
    n = 20_000
    x = hcat(Float64.(rand(Bool, n)), Float64.(rand(Bool, n)), randn(n))
    y = 3 .* x[:, 1] .+ 1.5 .* x[:, 2] .+ 0.1 .* randn(n)
    kw = (loss=:mse, nrounds=1, eta=1.0, lambda=0.0, L2=0.0, max_depth=2, nbins=16)

    mb = fit(EvoTreeRegressor(; kw..., gamma=0.0); x_train=x, y_train=y, verbosity=0)
    # gamma sits between a depth-1 node's share and the depth total, so dividing before the
    # gamma check instead of only in the stored value would reject the second depth
    mo = fit(EvoTreeRegressor(; kw..., gamma=8000.0, tree_type=:oblivious); x_train=x, y_train=y, verbosity=0)

    tree = mo.trees[1]
    @test sum(tree.split) == 3
    @test mb(x) == mo(x)

    ib = Dict(EvoTrees.importance(mb))
    io = Dict(EvoTrees.importance(mo))
    @test all(isapprox(io[k], ib[k]; atol=1e-5) for k in keys(ib))

    # the stored gains of a tree sum to its loss reduction, as they do for binary trees
    sse(p) = sum(abs2, y .- p)
    @test sum(tree.gain[tree.split]) ≈ sse(fill(mean(y), n)) - sse(mo(x)) rtol = 1e-4
end
