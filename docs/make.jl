using Documenter
using EvoTrees

push!(LOAD_PATH,"../src/")

pages = ["Home" => "index.md",
    "Examples" => "examples.md"]

makedocs(
    sitename="EvoTrees.jl",
    authors = "Jeremie Desgagne-Bouchard and contributors.",
    format=Documenter.HTML(),
    pages = pages,
    modules = [EvoTrees],)

deploydocs(repo="https://github.com/Evovest/EvoTrees.jl", 
    target="build",
    push_preview=false)