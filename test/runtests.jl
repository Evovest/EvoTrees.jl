using Statistics
using EvoTrees
using EvoTrees: predict
using Random
using Test

@testset "EvoTrees" begin

    @testset "Internal API" begin
        include("core.jl")
        include("monotonic.jl")
    end

    @testset "MLJ" begin
        include("MLJ.jl")
    end
end
