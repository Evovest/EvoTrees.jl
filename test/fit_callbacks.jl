struct CallbackSentinel <: Exception end

@testset "fit callbacks" begin
    rng = MersenneTwister(2026)
    x = randn(rng, Float32, 80, 3)
    y = Float32.(2 .* x[:, 1] .- x[:, 2])
    x_train, x_eval = x[1:60, :], x[61:end, :]
    y_train, y_eval = y[1:60], y[61:end]

    matrix_events = NamedTuple[]
    matrix_callback = function (model, logger, iteration)
        push!(
            matrix_events,
            (iteration=iteration, metric=logger[:metrics][end], rounds=model.info[:nrounds]),
        )
        return :ignored
    end
    matrix_params = EvoTreeRegressor(
        loss=:mse,
        metric=:mse,
        nrounds=4,
        max_depth=3,
        seed=7,
    )
    matrix_model = fit(
        matrix_params;
        x_train,
        y_train,
        x_eval,
        y_eval,
        callbacks=matrix_callback,
        verbosity=0,
    )
    @test getproperty.(matrix_events, :iteration) == 1:4
    @test getproperty.(matrix_events, :rounds) == 1:4
    @test all(isfinite, getproperty.(matrix_events, :metric))
    @test matrix_model.info[:nrounds] == 4

    table_train = (; a=x_train[:, 1], b=x_train[:, 2], c=x_train[:, 3], target=y_train)
    table_eval = (; a=x_eval[:, 1], b=x_eval[:, 2], c=x_eval[:, 3], target=y_eval)
    table_events = Tuple{Int,Float64}[]
    table_model = fit(
        matrix_params,
        table_train;
        target_name=:target,
        deval=table_eval,
        callbacks=((model, logger, iteration) ->
            push!(table_events, (iteration, logger[:metrics][end])),),
        verbosity=0,
    )
    @test first.(table_events) == 1:4
    @test table_model.info[:nrounds] == 4

    no_eval_events = Any[]
    no_eval_model = fit(
        EvoTreeRegressor(loss=:mse, nrounds=2, seed=8);
        x_train,
        y_train,
        callbacks=(model, logger, iteration) ->
            push!(no_eval_events, (logger, iteration)),
        verbosity=0,
    )
    @test no_eval_events == [(nothing, 1), (nothing, 2)]
    @test no_eval_model.info[:logger] === nothing

    @test_throws CallbackSentinel fit(
        EvoTreeRegressor(loss=:mse, nrounds=3, seed=9);
        x_train,
        y_train,
        callbacks=(model, logger, iteration) -> throw(CallbackSentinel()),
        verbosity=0,
    )

    early_events = Int[]
    constant_x = zeros(Float32, 30, 2)
    constant_y = ones(Float32, 30)
    early_model = fit(
        EvoTreeRegressor(
            loss=:mse,
            metric=:mse,
            nrounds=10,
            early_stopping_rounds=1,
            seed=10,
        );
        x_train=constant_x,
        y_train=constant_y,
        x_eval=constant_x,
        y_eval=constant_y,
        callbacks=(model, logger, iteration) -> begin
            @test logger[:iter_since_best] >= logger[:early_stopping_rounds]
            push!(early_events, iteration)
        end,
        verbosity=0,
    )
    @test early_events == [1]
    @test early_model.info[:nrounds] == 1
end
