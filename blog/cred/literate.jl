using Literate

filepath = joinpath(@__DIR__, "README.jl")
outputdir = @__DIR__
Literate.markdown(filepath, outputdir, flavor=Literate.CommonMarkFlavor())
