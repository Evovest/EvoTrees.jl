const DOCS_DIR = @__DIR__

qmd_files = [
    joinpath("tutorials", "offset-usage.qmd"),
]

for doc in qmd_files
    src = joinpath(DOCS_DIR, "quarto", doc)
    out = joinpath(DOCS_DIR, "src")
    @info "quarto render" doc output_dir = out
    run(`quarto render $src --output-dir $out`)
end
