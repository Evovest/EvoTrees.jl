using Statistics
using EvoTrees
using EvoTrees: fit, predict
using CategoricalArrays
using Tables
using Random
using Test

@testset "EvoTrees" begin

    @testset "Internal API" begin
        include("core.jl")
        include("oblivious.jl")
        include("tables.jl")
        include("monotonic.jl")
        include("missings.jl")
    end

    @testset "MLJ" begin
        include("MLJ.jl")
    end
end
