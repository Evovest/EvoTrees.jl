using Statistics
using EvoTrees
using Random
using Test

@testset "EvoTrees" begin

    @testset "Internal API" begin
        include("core.jl")
    end

    @testset "MLJ" begin
        include("MLJ.jl")
    end
end
