using CSV
using DataFrames
using PrettyTables
using Format: format

device = "gpu"

df = CSV.read(joinpath(@__DIR__, "regressor-$device.csv"), DataFrame)
df = df[:, Cols(Not(:device))]

# transform!(df, [:train_evo, :train_xgb, :infer_evo, :infer_xgb] .=> (x -> round.(x; sigdigits=3)) .=> [:train_evo, :train_xgb, :infer_evo, :infer_xgb])
transform!(df, [:train_evo, :train_xgb, :infer_evo, :infer_xgb] .=> (x -> format.(x; precision=2)) .=> [:train_evo, :train_xgb, :infer_evo, :infer_xgb])
transform!(df, :nobs => (x -> format.(x; autoscale=:metric)) => :nobs)

pretty_table(df; backend=Val(:markdown), show_subheader=false, alignment=:c)
