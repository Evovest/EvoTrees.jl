using Random
using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: tiedrank

using CUDA
using EvoTrees
using AWS: AWSCredentials, AWSConfig, @service
@service S3

aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

path = "share/data/year/year.csv"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
df = DataFrame(CSV.File(raw; header=false))
df_tot = copy(df)

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
train_idx = DataFrame(CSV.File(raw; header=false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
eval_idx = DataFrame(CSV.File(raw; header=false))[:, 1] .+ 1

target_name = "y"
rename!(df_tot, "Column1" => target_name)
feature_names = setdiff(names(df_tot), ["y", "w"])
df_tot.w .= 1.0

function norm(x::AbstractVector{T}) where {T}
    return (x .- mean(x)) ./ std(x)
end
transform!(df_tot, feature_names .=> norm .=> feature_names)

dtrain = df_tot[train_idx, :];
deval = df_tot[eval_idx, :];
dtest = df_tot[(end-51630+1):end, :];

config = EvoTreeRegressor(
    nrounds=2000,
    loss=:mae,
    metric=:mse,
    eta=0.1,
    nbins=32,
    min_weight=1,
    max_depth=8,
    L2=100,
    gamma=0,
    rowsample=0.5,
    colsample=0.9,
    early_stopping_rounds=50,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m = EvoTrees.fit(
    config,
    dtrain;
    feature_names,
    target_name,
    deval,
    print_every_n=100,
);

p_test = m(dtest);
mse_test = mean((p_test .- dtest.y) .^ 2)
@info "MSE - dtest" mse_test
