push!(LOAD_PATH, "../src/")

using Documenter
using EvoTrees

pages = [
    "Introduction" => "index.md",
    "Models" => "models.md",
    "API" => "api.md",
    "Examples - API" => "examples-API.md",
    "Examples - MLJ" => "examples-MLJ.md"]

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