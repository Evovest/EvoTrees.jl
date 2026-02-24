using CategoricalArrays
using DataFrames
using EvoTrees

nobs = 10_000
nfeats = 3
nlevels = 16
nbins = 16

df = DataFrame(rand(nobs, nfeats), :auto)
df.cat = rand(1:nlevels, nobs) |> categorical
df.y = randn(nobs)
length(unique(df.cat))
target_name="y"
feature_names = setdiff(names(df), [target_name])

config = EvoTreeRegressor(; nbins)

EvoTrees.fit(
    config,
    df;
    target_name,
    feature_names,
)
