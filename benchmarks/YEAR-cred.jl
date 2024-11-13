using Revise
using CSV
using DataFrames
using EvoTrees
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!

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

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

transform!(X, names(X) .=> percent_rank .=> names(X))
X = collect(Matrix{Float32}(X))
Y = Float32.(Y)

x_tot, y_tot = X[1:(end-51630), :], Y[1:(end-51630)]
x_test, y_test = X[(end-51630+1):end, :], Y[(end-51630+1):end]
x_train, x_eval = x_tot[train_idx, :], x_tot[eval_idx, :]
y_train, y_eval = y_tot[train_idx], y_tot[eval_idx]

config = EvoTreeRegressor(
    T=Float32,
    nrounds=3200,
    loss=:credV1,
    eta=0.1,
    nbins=128,
    min_weight=1,
    max_depth=7,
    lambda=0,
    gamma=0.1,
    rowsample=0.8,
    colsample=0.8,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m, logger = fit_evotree(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    early_stopping_rounds=100,
    print_every_n=100,
    metric=:mse,
    return_logger=true,
);
p_evo = m(x_test);
sort(p_evo)
mean((p_evo .- y_test) .^ 2) * std(Y_raw)^2
