using Test
using Statistics
using Random
using CategoricalArrays
using EvoTrees
using EvoTrees: fit, predict, predict_leaf_idx

# Expected raw score that shap values are measured from: the bias plus each tree's
# cover-weighted mean leaf value.
function shap_base(m)
    b = Float64(m.bias[1])
    for tree in m.trees, i in eachindex(tree.split)
        is_leaf = !tree.split[i] && (i == 1 || tree.split[i>>1])
        is_leaf && (b += tree.pred[1, i] * tree.w[i] / tree.w[1])
    end
    return b
end

# Raw (link scale) score of each row, from the leaves predict routes it to.
function raw_score(m, data)
    leaf = predict_leaf_idx(m, data)
    return [m.bias[1] + sum(Float64(m.trees[j].pred[1, leaf[i, j]]) for j in eachindex(m.trees)) for i in axes(leaf, 1)]
end

@testset "shap" begin

    Random.seed!(123)
    nobs = 2_000
    x1, x2 = randn(nobs), randn(nobs)
    lev = rand(["a", "b", "c", "d", "e"], nobs)
    effect = Dict("a" => -1.0, "b" => 0.5, "c" => 1.5, "d" => -0.5, "e" => 0.0)
    y = x1 .+ 0.5 .* x2 .+ [effect[l] for l in lev] .+ 0.1 .* randn(nobs)

    cases = [
        ("numeric", (x1=x1, x2=x2, y=y), :mse, :binary),
        ("categorical", (x1=x1, x2=x2, c=categorical(lev), y=y), :mse, :binary),
        ("ordered categorical", (x1=x1, x2=x2, c=categorical(lev; ordered=true), y=y), :mse, :binary),
        ("Bool", (x1=x1, x2=x2, c=lev .== "c", y=y), :mse, :binary),
        ("categorical, oblivious", (x1=x1, x2=x2, c=categorical(lev), y=y), :mse, :oblivious),
        ("categorical, logloss", (x1=x1, x2=x2, c=categorical(lev), y=Float64.(y .> median(y))), :logloss, :binary),
    ]

    @testset "values sum to the raw score: $name" for (name, data, loss, tree_type) in cases
        config = EvoTreeRegressor(; loss, tree_type, nrounds=10, max_depth=4, eta=0.3, nbins=16)
        feature_names = filter(!=(:y), collect(keys(data)))
        m = fit(config, data; target_name=:y, feature_names)
        S = EvoTrees.shap(m, data)
        @test size(S) == (nobs, length(feature_names))
        raw = raw_score(m, data)
        loss == :mse && @test raw ≈ predict(m, data) rtol = 1e-5
        # routing errors move single rows by the size of a leaf value, far above this
        @test maximum(abs.(raw .- (shap_base(m) .+ vec(sum(S; dims=2))))) < 1e-4
    end

end
