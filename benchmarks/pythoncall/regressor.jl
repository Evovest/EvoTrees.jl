using PythonCall
using Random
using Statistics
using BenchmarkTools

# Import Python modules
xgb = pyimport("xgboost")
np = pyimport("numpy")

# Parameters
nobs = 1_000_000
nfeats = 100
nrounds = 200
max_depth = 5
Random.seed!(123)

# Generate random data
x_train = rand(Float32, nobs, nfeats)
y_train = rand(Float32, nobs)

# Convert to numpy arrays
x_train_np = np.array(x_train)
y_train_np = np.array(y_train)

# Create DMatrix

dtrain = xgb.DMatrix(x_train_np, label=y_train_np)
# Create watchlist for evaluation

# Set XGBoost parameters
params = Dict(
    "objective" => "reg:squarederror",
    "max_depth" => 10,
    "eta" => 0.05,
    "tree_method" => "hist",  # Use "hist" for CPU
    "nthread" => 8,
    "verbosity" => 0,
    # "print_every_n" => 10,
    "subsample" => 0.5,
    "colsample_bytree" => 0.5,
    "max_bin" => 64,
    "device" => "gpu",
) |> pydict

# Warmup
booster = xgb.train(params, dtrain, num_boost_round=5, evals=pylist([(dtrain, "train")]), verbose_eval=10)

# Training speed test
@info "Training XGBoost..."
train_time = @elapsed booster = xgb.train(params, dtrain, num_boost_round=nrounds, evals=pylist([(dtrain, "train")]), verbose_eval=10)
@info "Train time (s)" train_time

# Print evaluation results
if !isnothing(evals_result) && hasproperty(evals_result, "items")
    evals_dict = Dict(evals_result.items())
    @info "Eval results" evals_dict
end

# Inference speed test
@info "Predicting XGBoost..."
pred_time = @elapsed preds = booster.predict(dtrain)
@info "Predict time (s)" pred_time

# Output mean prediction for sanity check
@info "Mean prediction" mean(preds)
# mean(pyconvert(Array, preds))
