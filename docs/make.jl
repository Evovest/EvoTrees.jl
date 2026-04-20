using Documenter
using DocumenterVitepress
using EvoTrees

pages = [
    "Overview" => "overview.md",
    "Models" => "models.md",
    "API" => [
        "Public" => "api.md",
        "Internals" => "internals.md"
    ],
    "Tutorials" => [
        "Regression - Boston" => "tutorials/regression-boston.md",
        "Logistic Regression - Titanic" => "tutorials/logistic-regression-titanic.md",
        "Classification - IRIS" => "tutorials/classification-iris.md",
        "Ranking - Yahoo! LTRC" => "tutorials/ranking-LTRC.md",
        "Credibility-based loss" => "tutorials/cred-loss.md",
        "Internal API" => "tutorials/examples-API.md",
        "MLJ API" => "tutorials/examples-MLJ.md",
        "Offset Usage" => "tutorials/offset-usage.md"]
]

makedocs(;
    sitename="EvoTrees.jl",
    authors="Jeremie Desgagne-Bouchard and contributors.",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/Evovest/EvoTrees.jl",
        devbranch="main",
        devurl="dev",
    ),
    pages,
    modules=[EvoTrees],
    warnonly=true
)

DocumenterVitepress.deploydocs(;
    repo="github.com/Evovest/EvoTrees.jl.git",
    target="build",
    branch="gh-pages",
    devbranch="main"
)
