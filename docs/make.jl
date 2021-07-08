using Documenter
using EvoTrees

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="EvoTrees",
    format=Documenter.HTML(),
    modules=[EvoTrees])

deploydocs(repo="https://github.com/Evovest/EvoTrees.jl", 
    target="build",
    push_preview=false)