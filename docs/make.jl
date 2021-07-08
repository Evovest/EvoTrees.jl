using Documenter
using EvoTrees

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="EvoTrees.jl",
    authors = "Jeremie Desgagne-Bouchard and contributors.",
    format=Documenter.HTML(),
    pages = ["Home" => "index.md"],)

deploydocs(repo="https://github.com/Evovest/EvoTrees.jl", 
    target="build",
    push_preview=false)