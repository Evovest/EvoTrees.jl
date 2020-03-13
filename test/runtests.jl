using Statistics
using Test
using EvoTrees

@testset "EvoTrees" begin

@info "Testing core"
include("core.jl")

@info "Testing MLJ"
include("MLJ.jl")

end
