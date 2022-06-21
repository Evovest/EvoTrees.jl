push!(LOAD_PATH, "../src/")

using Documenter
using EvoTrees

pages = [
    "Source" => "https://github.com/Evovest/EvoTrees.jl",
    "Introduction" => "index.md",
    "Examples" => "examples.md"]

makedocs(
    sitename="EvoTrees.jl",
    authors="Jeremie Desgagne-Bouchard and contributors.",
    format=Documenter.HTML(
        sidebar_sitename=false,
        edit_link = "main"
    ),
    pages=pages,
    modules=[EvoTrees]
)

deploydocs(repo="github.com/Evovest/EvoTrees.jl.git",
    target="build",
    devbranch="main")