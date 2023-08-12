using DataFrames
using Distributions
using EvoTrees
using LinearAlgebra
using GLM
using Random

δ = 1.0e-6
b = fill(1.0 - δ, 3, 3) + δ * I
z = zeros(3, 3)
y = fill(0.5, 3)
dist = MvNormal([
    b z z 0.8*y
    z b z y
    z z b 1.2*y
    0.8*y' y' 1.2*y' 1.0])
Random.seed!(1)
mat = rand(dist, 10_000);
df = DataFrame(transpose(mat), [string.("x", 1:9); "y"]);
target_name = "y"

#################################
# Tables API
#################################
config = EvoTreeRegressor(seed=123)
m1 = fit_evotree(config,
    df;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m1)

config = EvoTreeRegressor(seed=124)
m2 = fit_evotree(config,
    df;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m2)

# permuted tables doesn't return the same result - numerical rounding error?
df2 = df[!, 10:-1:1]
config = EvoTreeRegressor()
m3 = fit_evotree(config,
    df2;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m3)

# manual check on col permutations
config = EvoTreeRegressor(max_depth=4)
m1, cache1 = EvoTrees.init(config, df; target_name);
EvoTrees.grow_evotree!(m1, cache1, config, EvoTrees.CPU)
EvoTrees.importance(m1)

df2 = df[!, 10:-1:1];
config = EvoTreeRegressor(max_depth=4)
m2, cache2 = EvoTrees.init(config, df2; target_name);
EvoTrees.grow_evotree!(m2, cache2, config, EvoTrees.CPU)
EvoTrees.importance(m2)

all(cache1.x_bin .== cache2.x_bin[:, 9:-1:1])
all(cache1.edges .== cache2.edges[9:-1:1])
m1.trees[2]
m2.trees[2]

m1.trees[2].feat
m2.trees[2].feat

Int.(m1.trees[2].cond_bin)
Int.(m2.trees[2].cond_bin)


config = EvoTreeRegressor(nrounds=100, eta=0.05, colsample=1.0)
m3 = fit_evotree(config,
    df;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m3)

#################################
# Tables API
#################################
config = EvoTreeRegressor(colsample=0.5)
m1 = fit_evotree(config,
    df;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m1)

m2 = fit_evotree(config,
    df;
    target_name="y",
    verbosity=0);
EvoTrees.importance(m2)

#################################
# Matrix API
#################################
x_train = Matrix(mat[1:9, :]')
y_train = mat[10, :]

config = EvoTreeRegressor()
m1 = fit_evotree(config;
    x_train,
    y_train,
    verbosity=0);
EvoTrees.importance(m1)

m2 = fit_evotree(config;
    x_train,
    y_train,
    verbosity=0);
EvoTrees.importance(m2)

using GLM
x_train = Matrix(mat[1:9, :]')
y_train = mat[10, :]
lm(x_train, y_train)

#################################
# Matrix debug API
#################################
x_train = Matrix(mat[1:9, :]')
y_train = mat[10, :]

config = EvoTreeRegressor()
m1, cache1 = EvoTrees.init(config, x_train, y_train);
EvoTrees.grow_evotree!(m1, cache1, config, EvoTrees.CPU)
EvoTrees.importance(m1)

m2, cache2 = EvoTrees.init(config, x_train, y_train);
EvoTrees.grow_evotree!(m2, cache2, config, EvoTrees.CPU)
EvoTrees.importance(m2)

using MLJ
using EvoTrees
using MLJLinearModels
X, y = make_regression()
model = Stack(
    metalearner=LinearRegressor(),
    resampling=CV(nfolds=2),
    tree=EvoTreeRegressor()
)
mach = machine(model, X, y)
fit!(mach)
