using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using CairoMakie
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

# prepare a dataset
tree_type = :binary # binary/oblivious
_device = :cpu

Random.seed!(123)
features = rand(1000) .* 4 .- 1
x_train = reshape(features, (size(features)[1], 1))
# y_train = 0.5 .* features .^ 2 # deterministic

y_train = sin.(features) .* 0.5 .+ 0.5
y_train = logit(y_train) + randn(size(y_train)) ./ 2
y_train = sigmoid(y_train)

#################
# mse
#################
config = EvoTreeRegressor(;
    loss=:mse,
    nrounds=4,
    bagging_size=1,
    early_stopping_rounds=50,
    nbins=8,
    L2=0,
    eta=0.7,
    max_depth=3,
)
model = fit(
    config;
    x_train,
    y_train,
    print_every_n=25,
);
x_perm = sortperm(x_train[:, 1])
pred_mse = predict(model, x_train)[x_perm]

# animation settings
framerate = 1
iters = 1:4

itr = Observable(4)
pred_mse = @lift(predict(model, x_train; ntree_limit=$itr)[x_perm])

f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#c9ccd1",
    markersize=6)
lines!(ax,
    x_train[x_perm, 1],
    pred_mse,
    color="#26a671",
    linewidth=3,
    label="mse",
)
f
record(f, joinpath(@__DIR__, "boost-anim.gif"), iters; framerate=framerate) do iter
    itr[] = iter
end

using Plots: plot

for i in 1:4
    p = plot(model, i)
    save("model-plt-$i.svg", p)
end

# using Images
using FileIO
img1 = load("model-plt-1.png")
img2 = load("model-plt-2.png")
img3 = load("model-plt-3.png")
img4 = load("model-plt-4.png")
f = Figure()
ax11 = Axis(f[1, 1], aspect=DataAspect())
ax12 = Axis(f[1, 2], aspect=DataAspect())
ax21 = Axis(f[2, 1], aspect=DataAspect())
ax22 = Axis(f[2, 2], aspect=DataAspect())
hidedecorations!(ax11)
hidedecorations!(ax12)
hidedecorations!(ax21)
hidedecorations!(ax22)
hidespines!(ax11)
hidespines!(ax12)
hidespines!(ax21)
hidespines!(ax22)
image!(ax11, rotr90(img1))
image!(ax12, rotr90(img2))
image!(ax21, rotr90(img3))
image!(ax22, rotr90(img4))
f

ax_vec = [ax11, ax12, ax21, ax22]
imgs = [img1, img2, img3, img4]
record(f, "boost-anim-tree.gif", 1:4; framerate=framerate) do i
    image!(ax_vec[i], rotr90(imgs[i]))
end
