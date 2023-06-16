using Revise
using EvoTrees
using MLUtils
using CSV
using DataFrames
using Arrow
using CUDA
using Base.Iterators: partition
using Base.Threads: nthreads, @threads
using Tables

########################################
# create regular dataframe
########################################
nobs = Int(1e6)
nfeats = Int(100)
x_train = rand(nobs, nfeats)
y_train = rand(nobs)
df = DataFrame(x_train, :auto)
df[!, :y] = y_train
path = joinpath(@__DIR__, "..", "data", "arrow-df.arrow")
Arrow.write(path, df)

########################################
# read streaming data
########################################
path = joinpath(@__DIR__, "..", "data", "arrow-df.arrow")
@time dtrain = Arrow.Table(path);
@time dtrain = DataFrame(Arrow.Table(path));
@time dtrain = DataFrame(Arrow.Table(path), copycols=false);

function load_1()
    df = DataFrame(Arrow.Table(path), copycols=true)
    select!(df, [:x1, :x2, :x3, :x4, :x5])
    return df
end
function load_2()
    df = DataFrame(Arrow.Table(path), copycols=false)
    select!(df, [:x1, :x2, :x3, :x4, :x5])
    return df
end
@time df = load_1();
@time df = load_2();
