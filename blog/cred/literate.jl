using Literate

filepath = joinpath(@__DIR__, "README.jl")
outputdir = @__DIR__
Literate.markdown(filepath, outputdir; credit=false,
    execute=true,
    mdstrings=true,
    flavor=Literate.CommonMarkFlavor()
)
