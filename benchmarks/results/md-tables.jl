using CSV
using DataFrames
using PrettyTables
using Format: format

device = "cpu"

df = CSV.read(joinpath(@__DIR__, "regressor-$device.csv"), DataFrame)
df = df[:, Cols(Not(:device))]

df_xgb = CSV.read(joinpath(@__DIR__, "..", "pythoncall", "results", "regressor-xgb-$device.csv"), DataFrame)
df_xgb = df_xgb[:, Cols(Not(:device))]
df_xgb.max_depth .+= 1
leftjoin!(df, df_xgb, on=[:nobs, :nfeats, :max_depth])

transform!(df, [:train_evo, :train_xgb, :infer_evo, :infer_xgb] .=> (x -> format.(x; precision=2)) .=> [:train_evo, :train_xgb, :infer_evo, :infer_xgb])
# transform!(df, [:train_evo, :infer_evo] .=> (x -> format.(x; precision=2)) .=> [:train_evo, :infer_evo])
transform!(df, :nobs => (x -> format.(x; autoscale=:metric)) => :nobs)
pretty_table(df; backend=:markdown, show_first_column_label_only=true, alignment=:c)
