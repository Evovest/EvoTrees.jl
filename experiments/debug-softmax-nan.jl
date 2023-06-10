using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using Random: seed!

seed = 12
seed!(seed)
x_train = Array([
   sin.(1:1000) rand(1000)
   100 .* cos.(1:1000) rand(1000) .+ 1 
]);
y_train = repeat(1:2; inner = 1000);
params1 = EvoTreeClassifier(; T = Float32, max_depth=3, rowsample  = 0.5, nrounds = 200, eta = 1, rng=seed)
m = fit_evotree(params1; x_train, y_train)
preds = EvoTrees.predict(m, x_train)[:, 1]
sort(preds)

m.trees[end].pred