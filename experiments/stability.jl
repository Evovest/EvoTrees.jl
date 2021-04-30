
# using MLJ, EvoTrees, MLJScientificTypes
using MLJ, EvoTrees, CSV, DataFrames

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

#,types=Dict("Sex"=>CategoricalValue{String, UInt32})
# + tags=[]
df = DataFrame!(CSV.File("./data/train.csv",pool=0.1, missingstrings=[" "]))
categorical!(df,[cat_cols;target_col]);
describe(df,:eltype,:nunique, :nmissing)
dropmissing!(df);
describe(df,:eltype,:nunique, :nmissing)

X = df[!, all_feature_cols];
y = df[!,target_col];

mach_x = machine(ContinuousEncoder(), X)
fit!(mach_x)
X = MLJ.transform(mach_x, X)

X_mat = Array(X)

tree_model = EvoTreeClassifier(T=Float64, max_depth=6, nrounds=200, η = 0.2, colsample=0.3, metric=:mlogloss)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
x_train, y_train = X_mat[train, :], y[train]
x_test, y_test = X_mat[test, :], y[test]
@time model = fit_evotree(tree_model, x_train, y_train, print_every_n = 10);
pred_test = EvoTrees.predict(model, x_test)
minimum(pred_test)

tree_model = EvoTreeClassifier(max_depth=6, nrounds=2000,colsample=0.3)
mach = machine(tree_model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)
pred_test = MLJ.predict(mach, selectrows(X, test))

y_train_num = Float64.([y.level for y in y_train]) .- 1
tree_model = EvoTreeRegressor(T=Float32, loss=:logistic, metric=:logloss, max_depth=6, nrounds=200, η=0.2, colsample=0.3)
@time model = fit_evotree(tree_model, x_train, y_train_num, print_every_n = 10);
pred_test = EvoTrees.predict(model, x_test)
minimum(pred_test)
maximum(pred_test)