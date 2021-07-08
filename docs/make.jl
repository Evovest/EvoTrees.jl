using Documenter
using EvoTrees

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "EvoTrees",
    format   = Documenter.HTML(),
    modules  = [EvoTrees])