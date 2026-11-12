# create plots for student-mle experiments
# https://github.com/Evovest/EvoTrees.jl/blob/main/experiments/readme_plots.jl
# https://github.com/Evovest/EvoTrees.jl/blob/main/experiments/readme_plots-multi-target.jl
using DataFrames
using CairoMakie

using Statistics
using Random
using Distributions
using StatsBase: corspearman
using EvoTrees
using EvoTrees: fit, predict

_device = :cpu
tree_type = :binary
assets = joinpath(@__DIR__, "docs", "src", "assets", "student-mle")

if _device == :gpu
    using CUDA
    CUDA.functional()
    Base.get_extension(EvoTrees, :EvoTreesCUDAExt)
end 

###############################
# Data-generating process
#   y = μ(x₁) + σ(x₁)·ε,  ε has unit scale
#   x₂…x₅ are pure-noise features (gives the trees spurious splits to chase)
###############################
μ_fn(x) = 0.5 * sin(x)
σ_fn(x) = 0.2 + 0.2 * x            # heteroskedastic: 0.2 → 1.2

scenarios = [
    (name=:gaussian, ε=Normal()), # null case
    (name=:t3, ε=TDist(3)), # well-specified heavy tails
    (name=:contaminated, ε=MixtureModel([Normal(0, 1), Normal(0, 10)], [0.95, 0.05])),  # robustness hypothesis
]

function simulate(rng, n, ε)
    X = rand(rng, n, 5) .* 5
    μ = μ_fn.(X[:, 1])
    σ = σ_fn.(X[:, 1])
    y = μ .+ σ .* rand(rng, ε, n)
    return (; X, y, μ, σ)
end

###############################
# Models
###############################
train_kw = (
    nrounds=2000,
    early_stopping_rounds=50,
    nbins=64,
    eta=0.05,
    max_depth=5,
    min_weight=8.0,
    rowsample=0.5,
    colsample=1.0,
    L2=0.0,
    tree_type,
    device=_device,
)

# quantile of each family on the (μ, scale) heads
q_family(D) = (μ, s, p) -> μ .+ s .* quantile(D, p)

model_specs = [
    (label="gaussian_mle", ctor=EvoTreeMLE, kw=(loss=:gaussian_mle,), q=q_family(Normal())),
    (label="logistic_mle", ctor=EvoTreeMLE, kw=(loss=:logistic_mle,), q=q_family(Logistic())),
    [(label="student_mle ν=$ν", ctor=EvoTreeMLE, kw=(loss=:student_mle, nu=ν), q=q_family(TDist(ν)))
     for ν in (3.0, 5.0, 10.0)]...,
]

function fit_eval(spec, d_train, d_eval, d_test, seed)
    cfg = spec.ctor(; train_kw..., spec.kw..., seed)
    m = fit(cfg; x_train=d_train.X, y_train=d_train.y, x_eval=d_eval.X, y_eval=d_eval.y, verbosity=0)
    nt = max(m.info[:logger][:best_iter], 1)          # fit() keeps all trees; stop at best
    p = predict(m, d_test.X; device=_device, ntree_limit=nt)
    μ̂ = p isa AbstractVector ? p : p[:, 1]

    cov10 = cov90 = scale_ratio = NaN
    if !isnothing(spec.q)
        ŝ = p[:, 2]
        cov10 = mean(d_test.y .< spec.q(μ̂, ŝ, 0.1))
        cov90 = mean(d_test.y .< spec.q(μ̂, ŝ, 0.9))
        scale_ratio = median(ŝ ./ d_test.σ)          # ≈ 1 only when the family matches ε
    end

    row = (
        model=spec.label, seed, ntrees=nt,
        rmse_mu=sqrt(mean((μ̂ .- d_test.μ) .^ 2)),    # signal recovery vs the truth
        cor_mu=cor(μ̂, d_test.μ),
        pearson=cor(μ̂, d_test.y),                     # what we can measure on real data
        spearman=corspearman(μ̂, d_test.y),
        cov10, cov90, scale_ratio,
    )
    return row, m, nt
end

###############################
# Run: scenarios × seeds × models
###############################
nseeds = 5
n_train, n_eval, n_test = 20_000, 5_000, 50_000

results = DataFrame()
fitted = Dict()
for sc in scenarios, seed in 1:nseeds
    rng = Xoshiro(seed)
    d_train = simulate(rng, n_train, sc.ε)
    d_eval = simulate(rng, n_eval, sc.ε)
    d_test = simulate(rng, n_test, sc.ε)
    for spec in model_specs
        row, m, nt = fit_eval(spec, d_train, d_eval, d_test, seed)
        push!(results, (scenario=sc.name, row...))
        seed == 1 && (fitted[(sc.name, spec.label)] = (m, nt, spec))
    end
    @info "done" sc.name seed
end

###############################
# Summary: mean over seeds
###############################
# cov10 / cov90 means coverage under q10 / q90 respectively
metrics = [:rmse_mu, :cor_mu, :pearson, :spearman, :cov10, :cov90, :scale_ratio, :ntrees]
summary = combine(groupby(results, [:scenario, :model]),
    metrics .=> mean .=> metrics,
    [:rmse_mu, :spearman] .=> std .=> [:rmse_mu_sd, :spearman_sd])
for sc in scenarios
    println("\n== $(sc.name) ==")
    show(stdout, filter(:scenario => ==(sc.name), summary)[:, Not(:scenario)]; allrows=true, allcols=true)
end

###############################
# Plot
###############################
x_grid = range(0, 5; length=500)
X_grid = hcat(collect(x_grid), fill(2.5, length(x_grid), 4))

for sc in scenarios
    rng = Xoshiro(1)
    d = simulate(rng, 5_000, sc.ε)
    ylo, yhi = quantile(d.y, (0.01, 0.99))

    f = Figure(size=(1000, 420))
    ax1 = Axis(f[1, 1]; title="$(sc.name): location", xlabel="x₁", ylabel="y")
    ax2 = Axis(f[1, 2]; title="$(sc.name): q10 / q90", xlabel="x₁")
    for ax in (ax1, ax2)
        scatter!(ax, d.X[:, 1], d.y; color="#BBB", markersize=2)
        ylims!(ax, ylo, yhi)
    end

    lines!(ax1, x_grid, μ_fn.(x_grid); color=:black, linewidth=2, label="true μ")
    for (lbl, color) in (("gaussian_mle", "darkred"), ("student_mle ν=3.0", "darkgreen"), ("logistic_mle", "darkblue"))
        m, nt, _ = fitted[(sc.name, lbl)]
        p = predict(m, X_grid; device=_device, ntree_limit=nt)
        lines!(ax1, x_grid, p isa AbstractVector ? p : p[:, 1]; color, linewidth=1.5, label=lbl)
    end

    for p_ in (0.1, 0.9)
        lines!(ax2, x_grid, μ_fn.(x_grid) .+ σ_fn.(x_grid) .* quantile(sc.ε, p_);
            color=:black, linewidth=2, label=p_ == 0.1 ? "truth" : nothing)
    end
    for (lbl, color) in (("gaussian_mle", "darkred"), ("student_mle ν=3.0", "darkgreen"), ("logistic_mle", "darkblue"))
        m, nt, spec = fitted[(sc.name, lbl)]
        p = predict(m, X_grid; device=_device, ntree_limit=nt)
        for p_ in (0.1, 0.9)
            lines!(ax2, x_grid, spec.q(p[:, 1], p[:, 2], p_); color, linewidth=1.5,
                label=p_ == 0.1 ? lbl : nothing)
        end
    end

    Legend(f[2, 1], ax1; orientation=:horizontal, tellwidth=false)
    Legend(f[2, 2], ax2; orientation=:horizontal, tellwidth=false)
    save(joinpath(assets, "student-synthetic-$(sc.name)-$tree_type-$_device.png"), f)
end