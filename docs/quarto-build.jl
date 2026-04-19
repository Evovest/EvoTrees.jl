const DOCS_DIR = @__DIR__

qmd_files = [
    "tutorials/offset-usage.qmd",
]

for rel in qmd_files
    src = joinpath(DOCS_DIR, "quarto", rel)
    out = joinpath(DOCS_DIR, "src", dirname(rel))
    @info "quarto render" file=src output_dir=out
    run(`quarto render $src --output-dir $out`)
end
