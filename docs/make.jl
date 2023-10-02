push!(LOAD_PATH, "../src/")

using Documenter
using EvoTrees

pages = [
    "Introduction" => "index.md",
    "Models" => "models.md",
    "API" => "api.md",
    "Tutorials" => [
        "Regression - Boston" => "tutorials/regression-boston.md",
        "Logistic Regression - Titanic" => "tutorials/logistic-regression-titanic.md",
        "Classification - IRIS" => "tutorials/classification-iris.md",
        "Ranking - Yahoo! LTRC" => "tutorials/ranking-LTRC.md",
        "Internal API" => "tutorials/examples-API.md",
        "MLJ API" => "tutorials/examples-MLJ.md"]
]

makedocs(
    sitename="EvoTrees.jl",
    authors="Jeremie Desgagne-Bouchard and contributors.",
    format=Documenter.HTML(
        sidebar_sitename=false,
        edit_link="main",
        assets=["assets/style.css"]
    ),
    pages=pages,
    modules=[EvoTrees],
    warnonly=true
)

deploydocs(repo="github.com/Evovest/EvoTrees.jl.git",
    target="build",
    devbranch="main")