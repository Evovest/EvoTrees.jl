using EvoTrees
using DataFrames
using MLJ

@load XGBoostClassifier
xgb = MLJXGBoostInterface.XGBoostClassifier()
[xgb, xgb]

# @load EvoTreeClassifier
evo = EvoTreeClassifier()
[evo, evo]
