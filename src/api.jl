"""
    train(
        params::EvoTypes{L,T}, 
        dtrain::AbstractDataFrame;
        target_name,
        fnames_num=nothing,
        fnames_cat=nothing,
        w_name=nothing,
        offset_name=nothing,
        group_name=nothing,
        deval=nothing,
        metric=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1,
        return_logger=false)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` input data. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount and EvoTreeMLE.
- `dtrain::DataFrames`: DataFrame containing features and target variables. 

# Keyword arguments

- `target_name`: name of target variable. 
- `w_name`: vector of train weights of length `#observations`. If `nothing`, a vector of ones is assumed.
- `offset_name`: name of the offset variable.
- `group_name`: name of the grouping variable.
- `x_eval::Matrix`: evaluation data of size `[#observations, #features]`. 
- `y_eval::Vector`: vector of evaluation targets of length `#observations`.
- `w_eval::Vector`: vector of evaluation weights of length `#observations`. Defaults to `nothing` (assumes a vector of 1s).
- `offset_eval::VecOrMat`: evaluation data offset. Should match the size of the predictions.
- `metric`: The evaluation metric that wil be tracked on `x_eval`, `y_eval` and optionally `w_eval` / `offset_eval` data. 
    Supported metrics are: 
    - `:mse`: mean-squared error. Adapted for general regression models.
    - `:rmse`: root-mean-squared error (CPU only). Adapted for general regression models.
    - `:mae`: mean absolute error. Adapted for general regression models.
    - `:logloss`: Adapted for `:logistic` regression models.
    - `:mlogloss`: Multi-class cross entropy. Adapted to `EvoTreeClassifier` classification models. 
    - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
    - `:gamma`: Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
    - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
    - `:gaussian_mle`: Gaussian log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :gaussian_mle`. 
    - `:logistic_mle`: Logistic log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :logistic_mle`. 
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `fnames`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
- `return_logger::Bool = false`: if set to true (default), `fit_evotree` return a tuple `(m, logger)` where logger is a dict containing various tracking information.
"""
function train(
    params::EvoTypes{L,T},
    dtrain::AbstractDataFrame;
    target_name,
    fnames_num=nothing,
    fnames_cat=nothing,
    w_name=nothing,
    offset_name=nothing,
    group_name=nothing,
    deval=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    return_logger=false,
    device="cpu"
) where {L,T}

    verbosity == 1 && @info params

    # initialize model and cache
    if String(device) == "gpu"
        m, cache = init_gpu(params, dtrain; target_name, fnames_num, fnames_cat, w_name, offset_name, group_name)
    else
        m, cache = init(params, dtrain; target_name, fnames_num, fnames_cat, w_name, offset_name, group_name)
    end

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(deval)
    any_flag = !isnothing(metric) || !isnothing(deval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, both `metric` and `deval` must be provided."
    end
    if logging_flag
        cb = CallBack(params, m, deval; target_name, w_name, offset_name, metric, device)
        logger = init_logger(; T, metric, maximise=is_maximise(cb.feval), early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    m, logger = train_loop!(m, cache, params; logger, cb, print_every_n, verbosity)
    if String(device) == "gpu"
        GC.gc(true)
        CUDA.reclaim()
    end

    if return_logger
        return (m, logger)
    else
        return m
    end
end


"""
    train(params;
        x_train, y_train, w_train=nothing, offset_train=nothing;
        x_eval=nothing, y_eval=nothing, w_eval=nothing, offset_eval=nothing,
        early_stopping_rounds=9999,
        print_every_n=9999,
        verbosity=1)

Main training function. Performs model fitting given configuration `params`, `x_train`, `y_train` input data. 

# Arguments

- `params::EvoTypes`: configuration info providing hyper-paramters. `EvoTypes` comprises EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount and EvoTreeMLE.

# Keyword arguments

- `x_train::Matrix`: training data of size `[#observations, #features]`. 
- `y_train::Vector`: vector of train targets of length `#observations`.
- `w_train::Vector`: vector of train weights of length `#observations`. If `nothing`, a vector of ones is assumed.
- `offset_train::VecOrMat`: offset for the training data. Should match the size of the predictions.
- `x_eval::Matrix`: evaluation data of size `[#observations, #features]`. 
- `y_eval::Vector`: vector of evaluation targets of length `#observations`.
- `w_eval::Vector`: vector of evaluation weights of length `#observations`. Defaults to `nothing` (assumes a vector of 1s).
- `offset_eval::VecOrMat`: evaluation data offset. Should match the size of the predictions.
- `metric`: The evaluation metric that wil be tracked on `x_eval`, `y_eval` and optionally `w_eval` / `offset_eval` data. 
    Supported metrics are: 
    - `:mse`: mean-squared error. Adapted for general regression models.
    - `:rmse`: root-mean-squared error (CPU only). Adapted for general regression models.
    - `:mae`: mean absolute error. Adapted for general regression models.
    - `:logloss`: Adapted for `:logistic` regression models.
    - `:mlogloss`: Multi-class cross entropy. Adapted to `EvoTreeClassifier` classification models. 
    - `:poisson`: Poisson deviance. Adapted to `EvoTreeCount` count models.
    - `:gamma`: Gamma deviance. Adapted to regression problem on Gamma like, positively distributed targets.
    - `:tweedie`: Tweedie deviance. Adapted to regression problem on Tweedie like, positively distributed targets with probability mass at `y == 0`.
    - `:gaussian_mle`: Gaussian log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :gaussian_mle`. 
    - `:logistic_mle`: Logistic log-likelihood. Adapted to MLE when using `EvoTreeMLE` with `loss = :logistic_mle`. 
- `early_stopping_rounds::Integer`: number of consecutive rounds without metric improvement after which fitting in stopped. 
- `print_every_n`: sets at which frequency logging info should be printed. 
- `verbosity`: set to 1 to print logging info during training.
- `fnames`: the names of the `x_train` features. If provided, should be a vector of string with `length(fnames) = size(x_train, 2)`.
- `return_logger::Bool = false`: if set to true (default), `fit_evotree` return a tuple `(m, logger)` where logger is a dict containing various tracking information.
"""
function train(
    params::EvoTypes{L,T};
    x_train::Matrix,
    y_train,
    w_train=nothing,
    offset_train=nothing,
    x_eval=nothing,
    y_eval=nothing,
    w_eval=nothing,
    offset_eval=nothing,
    metric=nothing,
    early_stopping_rounds=9999,
    print_every_n=9999,
    verbosity=1,
    fnames=nothing,
    return_logger=false,
    device="cpu"
) where {L,T}

    verbosity == 1 && @info params

    # initialize model and cache
    if String(device) == "gpu"
        m, cache = init_gpu(params, x_train, y_train; fnames, w_train, offset_train)
    else
        m, cache = init(params, x_train, y_train; fnames, w_train, offset_train)
    end

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(x_eval) && !isnothing(y_eval)
    any_flag = !isnothing(metric) || !isnothing(x_eval) || !isnothing(y_eval)
    if !logging_flag && any_flag
        @warn "To track eval metric in logger, `metric`, `x_eval` and `y_eval` must all be provided."
    end
    if logging_flag
        cb = CallBack(params, m, x_eval, y_eval; w_eval, offset_eval, metric, device)
        logger = init_logger(; T, metric, maximise=is_maximise(cb.feval), early_stopping_rounds)
        cb(logger, 0, m.trees[end])
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger, cb = nothing, nothing
    end

    m, logger = train_loop!(m, cache, params; logger, cb, print_every_n, verbosity)
    if String(device) == "gpu"
        GC.gc(true)
        CUDA.reclaim()
    end

    if return_logger
        return (m, logger)
    else
        return m
    end
end

function train_loop!(m, cache, params::EvoTypes; logger=nothing, cb=nothing, print_every_n, verbosity)
    for i = 1:params.nrounds
        train!(m, cache, params)
        if !isnothing(logger)
            cb(logger, i, m.trees[end])
            if i % print_every_n == 0 && verbosity > 0
                @info "iter $i" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    return (m, logger)
end
