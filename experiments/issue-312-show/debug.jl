using EvoTrees
using DataFrames
using MLJBase

@load XGBoostClassifier
@load EvoTreeClassifier
xgb = XGBoostClassifier()
[xgb, xgb]  # compact form

(@load EvoTreeClassifier)()
evo = EvoTreeClassifier()
[evo, evo]
