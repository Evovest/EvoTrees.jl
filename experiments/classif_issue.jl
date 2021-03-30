using Revise
using EvoTrees
using DataFrames
using CSV
using MLJ

num_cols = [
    "ClientPeriod",
    "MonthlySpending",
    "TotalSpent"
];

cat_cols = [
    "Sex",
    "IsSeniorCitizen",
    "HasPartner",
    "HasChild",
    "HasPhoneService",
    "HasMultiplePhoneNumbers",
    "HasInternetService",
    "HasOnlineSecurityService",
    "HasOnlineBackup",
    "HasDeviceProtection",
    "HasTechSupportAccess",
    "HasOnlineTV",
    "HasMovieSubscription",
    "HasContractPhone",
    "IsBillingPaperless",
    "PaymentMethod"
];

all_feature_cols = [num_cols; cat_cols];
target_col = "Churn";

# ,types=Dict("Sex"=>CategoricalValue{String, UInt32})
# + tags=[]
df = DataFrame!(CSV.File("data/train.csv", pool=0.1, missingstrings=[" "]))
categorical!(df,[cat_cols;target_col]);
describe(df,:eltype,:nunique, :nmissing)
dropmissing!(df);
describe(df,:eltype,:nunique, :nmissing)

X_df = df[!, all_feature_cols];
y = df[!,target_col];

mach_x = machine(ContinuousEncoder(), X_df)
fit!(mach_x)
X = MLJ.transform(mach_x, X_df)

tree_model = EvoTreeClassifier(T=Float32, max_depth=9, λ=1, nrounds=2000, colsample=0.3, metric=:mlogloss)
@time fit_evotree(tree_model, Array(X), y, print_every_n=100);
tree_model = EvoTreeClassifier(T=Float64, max_depth=9, λ=1, nrounds=2000, colsample=0.3, metric=:mlogloss)
@time fit_evotree(tree_model, Array(X), y, print_every_n=100);

y_num = Float64.(y.refs) .- 1
tree_model = EvoTreeRegressor(T=Float32, loss=:logistic, max_depth=9, nrounds=2000, colsample=0.3, metric=:logloss)
@time fit_evotree(tree_model, Array(X), y_num, print_every_n=100);
tree_model = EvoTreeRegressor(T=Float64, loss=:logistic, max_depth=9, nrounds=2000, colsample=0.3, metric=:logloss)
@time fit_evotree(tree_model, Array(X), y_num, print_every_n=100);

mach = machine(tree_model, X, y)


using XGBoost

params_xgb = ["max_depth" => 8,
         "eta" => 0.1,
         "objective" => "multi:softmax",
         "subsample" => 1.0,
         "colsample_bytree" => 0.3,
         "tree_method" => "hist",
         "max_bin" => 64,
         "lambda" => 1.0,
         "num_class" => 2,
         "print_every_n" => 50]
metrics = ["mlogloss"]

Int.(y_num)
@time m_xgb = xgboost(Array(X), 2000, label=Int.(y_num), param=params_xgb, metrics=metrics, nthread=8, print_every_n=100, silent=0);