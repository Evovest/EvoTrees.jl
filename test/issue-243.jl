using JLD2, EvoTrees
data = JLD2.load("data/data.jld2")
x_train = data["X"]
y_train = data["y"]

config = EvoTrees.EvoTreeMLE(; nrounds=100, eta=0.05, min_weight=2)
model = fit(config, metric=:gaussian_mle, x_train=x_train, y_train=y_train, x_eval=x_train, y_eval=y_train, print_every_n=1)
EvoTrees.predict(model, x_train)
