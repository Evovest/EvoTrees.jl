using CSV
using DataFrames
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!
using EvoTrees
using EvoTrees: fit

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

path = "share/data/year/year.csv"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
df = DataFrame(CSV.File(raw, header=false))

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
train_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
eval_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

X = df[:, 2:end]
Y_raw = Float64.(df[:, 1])
Y = (Y_raw .- mean(Y_raw)) ./ std(Y_raw)

x_tot, y_tot = X[1:(end-51630), :], Y[1:(end-51630)]
x_test, y_test = Matrix(X[(end-51630+1):end, :]), Y[(end-51630+1):end]
x_train, x_eval = Matrix(x_tot[train_idx, :]), Matrix(x_tot[eval_idx, :])
y_train, y_eval = y_tot[train_idx], y_tot[eval_idx]

config = EvoTreeRegressor(
    nrounds=3000,
    loss=:cred_std,
    metric=:mse,
    eta=0.1,
    nbins=32,
    min_weight=1,
    max_depth=7,
    lambda=0,
    L2=0,
    gamma=0,
    rowsample=0.5,
    colsample=0.9,
    early_stopping_rounds=50,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=100,
);
p_evo = m(x_test);
mean((p_evo .- y_test) .^ 2) * std(Y_raw)^2
