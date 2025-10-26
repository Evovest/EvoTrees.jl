# LearnAPI design exploration

Explore design around LearnAPI. 

## Considerations

- End user simplicity
    - Should be easy and intuitive to perform simple task. There's been regularly comments about how MLJ design appears appealingly robust, but hard to get started. 
    - Example: recent Slack question on how to fit Titanic dataset.
- Performance friendly
    - Reliance on a ML toolkit shouldn't impair the performance compared to usage of source model API. 
    - For example, there should be no overhead induced by the tracking of an eval metric on boosted tree models, as it is currently the case (as there's no friendly mechanism for caching eval's predictions)
- Production friendly
    - ML interface should be friendly to production context. Ie. consider the focus on reproducibility, deployment of pipelines, minimized/compiles inference models, save/load, etc. 
- Well defined scope
    - Is LearnAPI looking to cover any kind of model mapping X to Y?
    - Covers the typical ML models (linear, GLM, MixedModels, Trees, GBT, RandomForest).
    - Covers NeuralNetworks (regardless if built upon Flux, Lux, or other)
    - Time-series models
    - Unsupervised models (PCA, autoencoder...)
    - Data Transformation / Pipelines (are also mapping of X -> Y)
- Minimal constraints on model developer
    - Integration with the Interface should be minimal and limit the imposition of syntactic and design choices.
    - As users and devs can come from very diverse background, the intersection of what is seen as obvious and common is posed to be limited. Each additional constraint is a potential new source of friction for adoption. 

## General proposition

Refers to models as the broad object that comprise all potentially relevant info, including hyper_params / config, trained params, cache, etc.

Only `fit!` and `predict` methods are required to be implemented. 
Could even be limited to only `fit!` is we encourage Model's overloading, such that `model(x) == predict(model, x)`.

```julia
model = MyModel(; kwargs)
fit!(model, data; kwargs...)
predict(model, data)
```

For extra features, provides an opt-in design similar to Tables:

```julia
LearnAPI.support_iterative(::Type{MyModel}) = true
LearnAPI.support_online(::Type{MyModel}) = false
```

## Difference with current MLJ

- Usage of a common sourcing for actions function
    - MMI defines `fit`, but not `fit!` uses downstream in the MLJ ecosystem which comes from `MLJBase <- StatsBase <- StatsAPI`.
    - Reliance on StatsBase/StatsAPI appears undesirable considering they don't act much like base Interface / APIs. StatsBase has a excessively large dependency base. StatsAPI scope spills with linear model structures.
    - LearnAPI minimal dependency is a desirable design.

- Drop of the verbosity positional argument
    - Is a somewhat arbitrary convention that limits intuitive function usage and flexibility in using dispatch for representing broad high-level concepts. (see below the proposal for 1-2-3 positional arguments meaning)

- Definition number of targets or classes upfront
    - Desirable as whether there's 1 or more targets, or 2 or more classes, can influence the type of the model to be instantiated. 

- By only having a mutating `fit!`, it eliminates the subjectivity in the nature of the values returned by `fit`, as in MMI's tuple `(fitted, cache, report)`.

## Model instantiation

Aligned with current MLJ practice: 

```julia
model = MyModel(; kwargs...)
```

## fit!

`fit!` is expected to be a mutating function as the fitting process will alter the instantiated model through update to the model parameters and associated data.

Basic required implementation is a 2 positional arguments plus kwargs:

```julia
LeanAPI.fit!(model::LearnAPI.Model, dtrain; verbosity, kwargs...)
```

The above is meant to facilitate flexibility in support of various forms of data inputs.
For example:

```julia
LearnAPI.fit!(m::MyModel, (x_train, y_train))
LearnAPI.fit!(m::MyModel, df::DataFrame)
```

Having a single position arg to refer to training data allows to support various styles for passing in data:  
- single dataframe holding both features and target variables, and optionally weight and offset. Identification of variables can be performed through keyword argument (ex: feature_names = fnames::Vector{String}).
- `(features_matrix, target_vector)` tuple.  
 a single argument position for training data
- data loader: as used in neural network, notably through MLUtils's DataLoader
- We can also imagine support for a `path` reference to an on-disk storage (ex. `*.arrow`), a database connector...

### Iteration

Training of iterative models typically involve the tracking of eval metric on out-of-sample data. 
This notably touch most prevalent Tabular ML models like XGBoost, LightGBM, CatBoost, EvoTrees as well as any neural network / deep learning model.

Models that can be trained iteratively are identified with:
```julia
LearnAPI.support_iterative(::Type{MyModel}) = true
```

A 3-positional arg `fit!` must be implemented:

```julia
LeanAPI.fit!(model::LearnAPI.Model, dtrain, deval; verbosity, kwargs...)
```

Example of `fit!` implementation to support Iterated training.
In such scenario, ne need to input new data, as the model is trained on initialized datasets (ex. binarized data, data loaders...).
Allows for LearnAPI higher level utility to perform adaptation to some mutable hyper-params during the training process. 

```julia
LearnAPI.fit!(m::EvoTree; kwargs...)
```

This 1 positional `fit!` can play the role of LearnAPI's `update!`. Its usage would be come in the following context of `dynamic` iterative training: 

```julia
m = MyModel()
init!(m, dtrain) # init!(m, dtrain, deval)
iter = IterativeConfig() # defines a hyper parameter update strategy, like learning rate decay
fit!(iter, m) # this function would internally alternate modifications to hyper-params and calls to fit!(m) 
```

### Online

To support online mechanism, an additional function appears needed to distinguish between a complete `fit!` over a provided dataset (which may perform multiple cycles) and an online mechanism which performs a single update 
based on new input data.


## Notes

There's no need to explicitly rule on whether the arguments that allow the instantiation of a model are `hyper-parameters`, `hyper-params`, `config`, `strategy`.
Any model developer can use their own prefer terminology, according to their own taste and domain preference without altering with usability of LearnAPI. 

It can essentially come down to:
```julia
m = MyModel()
fit!(m, data)
p = m(data)
```

For actual implementation, that opens the opportunity for model implementation to rely on immutable hyper-params structs if desired. 
If model is iterable, and some hyper-params can be altered during the fitting process, idea would be to have to implement methods like: `list_mutable_hyper_params`, `set_hyper_param`, etc.

In many applications, I'd argue that we don't want the original configuration (hyper-param) to be mutated, as by doing so, the trace is lost of how we got to the final state of the hypers. 
I think it's desirable for the config / hyper-params that define the model constructor to represent the original snapshot that allows to fully reproduce a fit. 

TBD: whether the above can add pain for implementation a hyper search algo? 
Since model instantiation should be cheap, and model initialization for iterative models like EvoTrees unavoidable, it seems reasonable to instantiate a new model each time. 
May need to have helper mode to specify what hyper-params are searchable (not necessarily the same as the immutable ones, the later being specific to iterative models).

## Examples

### EvoTrees

Support for features::Matrix / target::Vector input pair:

```julia
function fit!(m::EvoTree, dtrain::Tuple{Matrix,Vector}; w=nothing, kwargs...)
    if !is_initialized(m)
        init!(m, dtrain; w_train=w, kwargs...)
    end
    while m.cache[:nrounds] < m.config.max_nrounds
        grow_evotree!(m)
    end
    return nothing
end
```

Support for single df::DataFrame input (requires an input kwarg to specify target_name):

```julia
function fit!(m::EvoTree, dtrain; target_name, kwargs...)
    if !is_initialized(m)
        init!(m, dtrain; target_name, kwargs...)
    end
    while m.cache[:nrounds] < m.config.max_nrounds
        grow_evotree!(m)
    end
    return nothing
end
```

Example of `fit!` implementation to support Iterated training.

```julia
function fit!(m::EvoTree; kwargs...)
    if is_initialized(m)
        while m.cache[:nrounds] < m.config.max_nrounds
            grow_evotree!(m)
        end
    end
    return nothing
end
```


Internal model struct:

```julia
struct EvoTreeRegressor <: MMI.Deterministic
    config::Config
    params::Params
    cache::Dict{Symbol,Any}
end

function EvoTreeRegressor(; kw...)
    config = Config(; kw...)
    params = Params(config)
    cache = Dict{Symbol,Any}(:is_initialized => false)
    return EvoTreeRegressor(config, params, cache)
end
```

Ideally, would only need a single constructor: `EvoTree(; kwargs...)`. Distinction between various flavors of `EvoTrees` could be handled through its parametric type. 
Ex:

```julia
LearnAPI.is_regressor(::Type{EvoTrees{MSE}}) = true
LearnAPI.is_regressor(::Type{EvoTrees{LogLoss}}) = true
LearnAPI.is_classifier(::Type{EvoTrees{MSE}}) = false
LearnAPI.is_classifier(::Type{EvoTrees{LogLoss}}) = true
```

### Linear

### Flux

### XGBoost

## LearnAPI review

Notes regarding the above design and the one from latesst [LearnAPI](https://juliaai.github.io/LearnAPI.jl/dev/)

### predict

Could the 3-arg `predict(model, LiteralTarget(), Xnew)` be changed to `predict(model, Xnew, proxy = LiteralTarget())`?

I think from that having `predict(m, x_new)` is an expected behavior for the `predict` function. 
By supporting this 2-arg version (where Literal Target is assumed), it likely covers a good share of use cases without adding compexity to the user, having to figure what are those target proxies are.

### Obs

Regarding [obs](https://juliaai.github.io/LearnAPI.jl/dev/obs/#Reference), I'm not clear about their role. 
Is it more about specifying the ingestion rules for various data input forms? Or is it more about acting as a preprocessing/initialization step?

I'm not clear it's desirable to have the full data representation (train + eval) within a common `data` container, which seems the approach taken here by LearnAPI. 
For instance, when performing cross-validations, there may be more extensive data preprocessing involved, potentially of the form `preproc = fit_preproc(dtrain); predict(preproc, eval_data)`.
So for sake of transparency and flexibility of the workflow, I see benefits in keeping the data splitting and preprocessing external to the model fitting process. 

How would iterative model be handled? In particular, how would the `preds` from eval data be store in order to provide efficient early stopping mechanism?

### Algo + fit vs fit!

LearnAPI required function implementations: 
 - predict
 - obsfit
 - obs (optional)

And structs:
- Algo: Ridge, EvoTreeRegressor
- AlgoFitData / Model: RidgeFitted, EvoTree
- AlgoData (optional): ex RidgeFitData, EvoTree.Cache?

EvoML required function implementations:
 - predict
 - fit!

And structs:
- Model: Ridge / EvoTree

### General design considerations

I'm wondering whether merging the notion of Algorithm/Config/Hyper-params and fitted model into a single entity `model` could address the opiniated debate on how to name each of these things. 
For example: `model = EvoTreeRegressor(); fit!(model, data...)`.
It may more likely render the model's struct to be mutable, but it doesn't bare the opportunity for that model struct to hold an immutable struct for the core params if needed. 
The `minimize` function seems like nice general functionality to handle situations where a model fitting procedure generates lots of cached / scratch data.

Other positives of the latest LearnAPI proposal: 
- no need to make Algo a subtype (eg: <: Deterministic). The traits approach looks like going in the right direction. 
- Dropping the arbitrary distinction between supervised & non-supervised. Will likely ease the integration of data preprocessing in the framework as a form of inference of from some self-supervised model.


## Discussion takeaways - 2023-12-07

- Makes sense to change LearnAPI's `predict` args order to `predict(m, x, proxy = LiteralTarget())`
- The `obs` and `obsfit` may not be necessary, at least not if the input data format are actually supported by the algorithm.
    - Post-note: should there be a more explicit definition of the type of data supported by the LearnAPI ecosystem? Ie. should it be limited to data where `Tables.istable() == true`?
    - Need to clarify the support for non standard use cases where there are a mixture of floats and categorical features (could be simply handled as a Table?).
- Having separate structs for a model's configuration/hyper-params/algo vs the fitted model is favored. As discussed in Slack threads, it still leaves lots of flexibility on how the fitted model is defined.
- TODO: clarify how to deal with iterative model, both EvoTrees/XGBoost and NeuralNetwork. Need a design proposal for IterativeModel to avoid reinventing the wheel for all algo on their early stopping support, while accounting for boosted model specifics where the eval/out-of-sample prediction need to be cached to avoid overhead (don't want to predict from scratch for each new layer of the stack).
