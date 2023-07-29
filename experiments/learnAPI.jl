using EvoTrees

module LearnAPI

abstract type Config end
abstract type Learner end
abstract type Model end

function fit(config::Config; kwargs...)
    return nothing
end
function fit(config::Config, data; kwargs...)
    return nothing
end
function init(config::Config, data; kwargs...)
    return nothing
end
# function fit!(learner::Learner)
#     return nothing
# end

function predict(model::Model, x)
    return x
end
function predict!(p, model::Model, x)
    return nothing
end

function isiterative(m) end

end #module

struct EvoLearner
    params
end

# 1 args fit: all needed supplemental info passed through kwargs: risk of having fragmentation of naming convention, hard to follow
m = LearnAPI.fit(config::Config; kwargs)
m = LearnAPI.fit(config::EvoTrees.EvoTypes; x_train=xt, y_train=yt)
m = LearnAPI.fit(config::EvoTrees.EvoTypes; x_train=xt, y_train=yt, x_eval=xe, y_eval=ye)

# 2 args fit: forces the notion of input data on which training is performed. May facilitates dispatch/specialisation on various supported data typees
m = LearnAPI.fit(config::Config, data; kwargs)
m = LearnAPI.fit(config::EvoTrees.EvoTypes, (x_train, y_train))
m = LearnAPI.fit(config::EvoTrees.EvoTypes, (x_train, y_train); x_eval=xe, y_eval=ye)
m = LearnAPI.fit(config::EvoTrees.EvoTypes, df::DataFrame)

# Iterative models
import .LearnAPI: isiterative
LearnAPI.isiterative(m::EvoTree) = true

# 2 args model initialization
# Here a EvoTreeLearner is returned: a comprehensive struct that includes the config, the model, and cache/state
m = LearnAPI.init(config::Config, data::DataFrame; kwargs)
m = LearnAPI.init(config::EvoTrees.EvoTypes, df::DataFrame; x_eval=xe, y_eval=ye)

LearnAPI.fit!(m::EvoTree)
LearnAPI.fit!(m::EvoTree, data)

# LearnAPI.fit!(m, config::EvoTrees.EvoTypes; kwargs)
LearnAPI.predict(m::EvoTrees.EvoTypes, x)

config = EvoTreeRegressor()
# m, cache = LearnAPI.init()

# should be possible to have model that specify feature treatment upfront at the Config level?
# Or rather have those passed at the fitted level?
