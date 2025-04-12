
using EvoTrees
using DataFrames
using CategoricalArrays
using Statistics
using CairoMakie

n_AC = 10000
y_AC = randn(n_AC)
y_AC .= (y_AC .- mean(y_AC)) ./ std(y_AC)
y_AC .= y_AC .* 0.1 .- 0.25
df_AC = DataFrame(x1=fill("A", n_AC), x2=fill("C", n_AC), y=y_AC)

n_BC = 9000
y_BC = randn(n_BC)
y_BC .= (y_BC .- mean(y_BC)) ./ std(y_BC)
y_BC .= y_BC .* 0.1 .+ 0.25
df_BC = DataFrame(x1=fill("B", n_BC), x2=fill("C", n_BC), y=y_BC)

n_BD = 1000
y_BD = randn(n_BD)
y_BD .= (y_BD .- mean(y_BD)) ./ std(y_BD)
y_BD .= y_BD .* 0.2 .+ 1.5
df_BD = DataFrame(x1=fill("B", n_BD), x2=fill("D", n_BD), y=y_BD)

dtrain = vcat(df_AC, df_BC, df_BD)
transform!(dtrain, [:x1, :x2] .=> categorical .=> [:x1, :x2])
transform!(dtrain, :y => (y -> (y .- mean(y)) ./ std(y)) => :y)
@info mean(dtrain.y)

config = EvoTreeRegressor(loss=:mse, nrounds=1, max_depth=2)
model_mse = EvoTrees.fit(config, dtrain; target_name="y")
@info model_mse.trees[2]

config = EvoTreeRegressor(loss=:cred_std, nrounds=1, max_depth=2)
model_std = EvoTrees.fit(config, dtrain; target_name="y")
@info model_std.trees[2]

x1_A = dtrain[dtrain.x1.=="A", :y]
x1_B = dtrain[dtrain.x1.=="B", :y]
f = Figure()
ax = Axis(f[1, 1];
    title="feature: x1",
    subtitle=
    """
    A - mean: $(round(mean(x1_A); digits=3)) | std: $(round(std(x1_A); digits=3))
    B - mean: $(round(mean(x1_B); digits=3)) | std: $(round(std(x1_B); digits=3))
    """
)
density!(ax, x1_A; color="#4571a5CC", label="A")
density!(ax, x1_B; color="#26a671CC", label="B")
Legend(f[2, 1], ax, orientation=:horizontal)
f
save(joinpath(@__DIR__, "assets", "dist-mse-cred-x1.png"), f);#hide

x2_C = dtrain[dtrain.x2.=="C", :y]
x2_D = dtrain[dtrain.x2.=="D", :y]
f = Figure()
ax = Axis(f[1, 1];
    title="feature: x2",
    subtitle=
    """
    C - mean: $(round(mean(x2_C); digits=3)) | std: $(round(std(x2_C); digits=3))
    D - mean $(round(mean(x2_D); digits=3)) | std: $(round(std(x2_D); digits=3))
    """
)
density!(ax, x2_C; color="#4571a5CC", label="C")
density!(ax, x2_D; color="#26a671CC", label="D")
Legend(f[2, 1], ax, orientation=:horizontal)
f
save(joinpath(@__DIR__, "assets", "dist-mse-cred-x2.png"), f);#hide
