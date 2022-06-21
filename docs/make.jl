push!(LOAD_PATH, "../src/")

using Documenter
using EvoTrees

pages = [
    "Introduction" => "index.md",
    "Examples" => "examples.md",
    "MLJ" => "MLJ.md"]

makedocs(
    sitename="EvoTrees.jl",
    authors="Jeremie Desgagne-Bouchard and contributors.",
    format=Documenter.HTML(
        sidebar_sitename=false,
        edit_link = "main",
        assets = ["assets/style.css"]
    ),
    pages=pages,
    modules=[EvoTrees]
)

deploydocs(repo="github.com/Evovest/EvoTrees.jl.git",
    target="build",
    devbranch="main")