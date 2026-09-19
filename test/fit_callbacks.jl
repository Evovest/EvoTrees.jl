struct FitCallbackStop <: Exception end

@testset "Per-round fit callbacks" begin
    rng = MersenneTwister(2026)
    x = randn(rng, Float32, 80, 3)
    y = @. 2f0 * x[:, 1] - x[:, 2]
    train = (; a=x[1:60, 1], b=x[1:60, 2], c=x[1:60, 3], target=y[1:60])
    eval = (; a=x[61:80, 1], b=x[61:80, 2], c=x[61:80, 3], target=y[61:80])
    for table in (false, true), bagging in (1, 3)
        params = EvoTreeRegressor(nrounds=4, max_depth=3, seed=7, bagging_size=bagging)
        runfit = callbacks -> table ?
            fit(deepcopy(params), train; target_name=:target, deval=eval, callbacks, verbosity=0) :
            fit(deepcopy(params); x_train=x[1:60, :], y_train=y[1:60],
                x_eval=x[61:80, :], y_eval=y[61:80], callbacks, verbosity=0)
        events = []
        order = Tuple{Int,Symbol}[]
        first = (model, logger, iteration) -> begin
            push!(events, (iteration, model.info[:nrounds], length(model.trees), logger[:metrics][end]))
            push!(order, (iteration, :first))
            :ignored
        end
        second = (_, _, iteration) -> push!(order, (iteration, :second))
        model = runfit((first, second))
        @test getindex.(events, 1) == 1:4
        @test getindex.(events, 2) == 1:4
        @test getindex.(events, 3) == bagging .* (1:4)
        @test all(isfinite, getindex.(events, 4))
        @test order == [(i, label) for i in 1:4 for label in (:first, :second)]
        @test predict(model, x) == predict(runfit(()), x)
        @test predict(model, x) == predict(runfit(nothing), x)
        @test_throws FitCallbackStop runfit((_, _, _) -> throw(FitCallbackStop()))
    end
    events = []
    fit(EvoTreeRegressor(nrounds=2, seed=7); x_train=x, y_train=y,
        callbacks=(_, logger, i) -> push!(events, (logger, i)), verbosity=0)
    @test events == [(nothing, 1), (nothing, 2)]

    stopped = Int[]
    @test_throws FitCallbackStop fit(EvoTreeRegressor(nrounds=10, early_stopping_rounds=1);
        x_train=zeros(Float32, 30, 2), y_train=ones(Float32, 30),
        x_eval=zeros(Float32, 30, 2), y_eval=ones(Float32, 30), verbosity=0,
        callbacks=(_, logger, i) -> begin
            @test logger[:iter_since_best] >= logger[:early_stopping_rounds]
            push!(stopped, i)
            throw(FitCallbackStop())
        end)
    @test stopped == [1]
end
