using Revise
using EvoTrees
using MLUtils
using CSV
using DataFrames
using Arrow
using CUDA
using Base.Iterators: partition
using Base.Threads: nthreads, @threads

########################################
# create streaming data
########################################
nobs = Int(1e3)
nfeats = Int(10)
x_train = rand(nobs, nfeats)
y_train = rand(nobs)

df = DataFrame(x_train, :auto)
df[!, :y] = y_train

# split into nthreads blocks
bs = cld(nobs, nthreads())
parts = partition(1:nobs, bs)
p = Tables.partitioner([view(df, p, :) for p in parts])

path = joinpath(@__DIR__, "..", "data", "arrow-stream.arrow")
Arrow.write(path, p)

########################################
# read streaming data
########################################
using Tables
path = joinpath(@__DIR__, "..", "data", "arrow-stream.arrow")
@time dtrain = Arrow.Stream(path);
for d in dtrain
    # @info typeof(d)
    # @info "summary: " Tables.nrow(d)
    x_train = DataFrame(d)
    nobs = size(x_train, 1)
    is_in = zeros(UInt32, nobs)
    is_out = zeros(UInt32, nobs)
    mask = zeros(UInt8, nobs)
end
