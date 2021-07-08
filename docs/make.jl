using Documenter
using EvoTrees

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="EvoTrees.jl",
    format=Documenter.HTML())

deploydocs(repo="https://github.com/Evovest/EvoTrees.jl", 
    target="build",
    push_preview=false)