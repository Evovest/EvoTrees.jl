push!(LOAD_PATH, "../src/")

using Documenter
using EvoTrees

pages = [
    "Home" => "index.md",
    "Examples" => "examples.md"]

makedocs(
    sitename = "EvoTrees.jl",
    authors = "Jeremie Desgagne-Bouchard and contributors.",
    format = Documenter.HTML(),
    pages = pages,
    modules = [EvoTrees],)

deploydocs(repo = "github.com/Evovest/EvoTrees.jl.git",
    target = "build",
    devbranch = "main")