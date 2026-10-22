using Test
using Statistics
using Random
using EvoTrees
using EvoTrees: fit, predict, predict_leaf_idx

@testset "predict_leaf_idx" begin

    nobs = 1_000
    nfeats = 5
    Random.seed!(123)
    x = rand(nobs, nfeats)
    y = x[:, 1] .* 2 .+ sin.(x[:, 2] .* 5) .+ 0.1 .* randn(nobs)

    config = EvoTreeRegressor(nrounds=10, max_depth=4, eta=0.2)
    m = fit(config; x_train=x, y_train=y)

    leaf_idx = predict_leaf_idx(m, x)

    @testset "shape and type" begin
        @test size(leaf_idx) == (nobs, length(m.trees))
        @test eltype(leaf_idx) == UInt32
    end

    @testset "indices point to actual leaves" begin
        for (j, tree) in enumerate(m.trees)
            idx = leaf_idx[:, j]
            @test all(1 .<= idx .<= length(tree.split))
            # A returned node must be a leaf, never an internal split node.
            @test !any(tree.split[i] for i in idx)
        end
    end

    @testset "agrees with an independent traversal" begin
        feattypes = m.info[:feattypes]
        x_bin = EvoTrees.binarize(x; feature_names=m.info[:feature_names], edges=m.info[:edges])
        for (j, tree) in enumerate(m.trees)
            for i in (1, 7, 123, nobs)
                nid = 1
                while tree.split[nid]
                    feat = tree.feat[nid]
                    cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] :
                           x_bin[i, feat] == tree.cond_bin[nid]
                    nid = nid << 1 + !cond
                end
                @test leaf_idx[i, j] == nid
            end
        end
    end

    @testset "consistent with predict" begin
        # Observations reaching the same leaf of every tree must receive the same prediction.
        p = predict(m, x)
        groups = Dict{Vector{UInt32},Vector{Int}}()
        for i in 1:nobs
            push!(get!(groups, leaf_idx[i, :], Int[]), i)
        end
        @test length(groups) > 1
        for is in values(groups)
            @test all(p[is] .≈ p[is[1]])
        end
    end

    @testset "ntree_limit" begin
        @test size(predict_leaf_idx(m, x; ntree_limit=3)) == (nobs, 3)
        @test predict_leaf_idx(m, x; ntree_limit=3) == leaf_idx[:, 1:3]
        @test size(predict_leaf_idx(m, x; ntree_limit=0)) == (nobs, 0)
        @test_throws ErrorException predict_leaf_idx(m, x; ntree_limit=length(m.trees) + 1)
    end

    @testset "max_depth=1 is one split" begin
        m1 = fit(EvoTreeRegressor(nrounds=3, max_depth=1, eta=1.0); x_train=x, y_train=y)
        @test all(length(tree.split) == 3 for tree in m1.trees)
        @test all(tree.split[1] for tree in m1.trees)
        @test all(in((2, 3)), predict_leaf_idx(m1, x))
    end

    @testset "tables input" begin
        feature_names = ["x$j" for j in 1:nfeats]
        dtrain = (; (Symbol("x$j") => x[:, j] for j in 1:nfeats)..., y=y)
        m_nt = fit(
            EvoTreeRegressor(nrounds=10, max_depth=4, eta=0.2),
            dtrain;
            target_name="y", feature_names,
        )
        idx_nt = predict_leaf_idx(m_nt, dtrain)
        @test size(idx_nt) == (nobs, length(m_nt.trees))
        # A table and the equivalent matrix must yield identical leaf assignments.
        @test idx_nt == leaf_idx
    end
end
