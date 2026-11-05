using Statistics
using Random
using Distributions
using DataFrames
using StatsBase: corspearman
using EvoTrees
using EvoTrees: fit, predict
using CairoMakie

_device = :cpu
tree_type = :binary
assets = joinpath(@__DIR__, "docs", "src", "assets", "student-mle")

# goal: student-mle's σ is not really the standard deviation. 
# student-mle's mu and sigma can be pull far off by extreme values, the median and MAD can't. 

###############################
# Switchable Student init
#   Overrides EvoTrees._student_init (used by init for StudentMLE on CPU and GPU).
#   :median/:mad reproduces the current branch exactly; :mean/:sd is gaussian_mle's init.
###############################
const STUDENT_INIT = Ref((loc=:median, scale=:mad))

@eval EvoTrees function _student_init(y::AbstractVector{T}, ν) where {T}
    mode = Main.STUDENT_INIT[]
    loc = mode.loc == :mean ? mean(y) : median(y)
    if mode.scale == :sd
        σ0 = std(y)
    else
        m = median(y)
        mad = median(abs.(y .- m))
        sd = length(y) > 1 ? std(y) : zero(T)
        σ0 = mad > 0 ? mad / Distributions.quantile(Distributions.TDist(ν), 0.75) :
             sd > 0 ? sd : one(T)
    end
    return T(loc), T(log(σ0))
end

###############################
# DGP: y = μ(x₁) + σ(x₁)·ε, x₂…x₅ noise features
###############################
# Gross errors: replace a fraction p of y with ±val (bad ticks / data errors),
# independent of x and σ(x). Mutates y in place.
function gross_errors(p, val; onesided=false)
    return (rng, y) -> begin
        bad = findall(rand(rng, length(y)) .< p)
        signs = onesided ? ones(length(bad)) : rand(rng, (-1.0, 1.0), length(bad))
        y[bad] .= val .* signs
        y
    end
end

scenarios = [
    # reference: heavy tails, no errors (same as before)
    (name=:t3, draw=(rng, n) -> rand(rng, TDist(3), n), corrupt=nothing),
    # 1% at ±1000: mean OK, sd ≈ 100 → tests the SCALE init
    (name=:gross_sym, draw=(rng, n) -> rand(rng, TDist(3), n), corrupt=gross_errors(0.01, 1000.0)),
    # 1% at +1000: mean shifted by ≈ +10 AND sd ≈ 100 → tests LOCATION + SCALE
    (name=:gross_onesided, draw=(rng, n) -> rand(rng, TDist(3), n), corrupt=gross_errors(0.01, 1000.0; onesided=true)),
    # Cauchy (ν = 1): variance infinite, mean undefined → sd init depends on the single largest draw
    (name=:cauchy, draw=(rng, n) -> rand(rng, TDist(1), n), corrupt=nothing),
]

function simulate(rng, n, sc; clean=false)
    X = rand(rng, n, 5) .* 5
    μ = μ_fn.(X[:, 1])
    y = μ .+ σ_fn.(X[:, 1]) .* sc.draw(rng, n)
    !clean && !isnothing(sc.corrupt) && sc.corrupt(rng, y)
    return (; X, y, μ)
end

###############################
# Variants
###############################
ν_fit = 4.0
variants = [
    (label="student mean/sd", loss=:student_mle, init=(loc=:mean, scale=:sd)), 
    (label="student mean/mad", loss=:student_mle, init=(loc=:mean, scale=:mad)),
    (label="student median/sd", loss=:student_mle, init=(loc=:median, scale=:sd)),
    (label="student median/mad", loss=:student_mle, init=(loc=:median, scale=:mad)), # current default
]
ref_label = "student median/mad"

nrounds = 1000
train_kw = (
    nrounds,
    early_stopping_rounds=10_000,    # never stop: we want the whole path; best_iter is still logged
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

# log-spaced checkpoints along the boosting path
checkpoints = unique(round.(Int, exp.(range(0, log(nrounds); length=30))))

function run_variant(v, d_train, d_eval, d_test, seed)
    !isnothing(v.init) && (STUDENT_INIT[] = v.init)
    kw = v.loss == :student_mle ? (nu=ν_fit,) : (;)
    cfg = EvoTreeMLE(; train_kw..., loss=v.loss, kw..., seed)
    m = fit(cfg; x_train=d_train.X, y_train=d_train.y, x_eval=d_eval.X, y_eval=d_eval.y, verbosity=0)
    STUDENT_INIT[] = (loc=:median, scale=:mad)      # restore default

    score(nt) = begin
        μ̂ = predict(m, d_test.X; device=_device, ntree_limit=nt)[:, 1]
        (rmse_mu=sqrt(mean((μ̂ .- d_test.μ) .^ 2)), spearman=corspearman(μ̂, d_test.y))
    end

    best = max(m.info[:logger][:best_iter], 1)
    path = [(ntrees=nt, score(nt)...) for nt in checkpoints if nt <= length(m.trees)]
    at_best = score(best)
    return (; bias=Float64.(m.bias[1:2]), best, at_best, path)
end

###############################
# Run
###############################
nseeds = 5
n_train, n_eval, n_test = 20_000, 5_000, 50_000

final = DataFrame()    # one row per scenario × seed × variant
paths = DataFrame()    # one row per scenario × seed × variant × checkpoint

for sc in scenarios, seed in 1:nseeds
    rng = Xoshiro(seed)
    d_train = simulate(rng, n_train, sc)
    d_eval = simulate(rng, n_eval, sc)
    d_test = simulate(rng, n_test, sc; clean=true)
    for v in variants
        r = run_variant(v, d_train, d_eval, d_test, seed)
        # iterations needed to get within 5% of this run's best rmse_mu on the path
        rmin = minimum(p.rmse_mu for p in r.path)
        iters_to_5pct = first(p.ntrees for p in r.path if p.rmse_mu <= 1.05 * rmin)
        push!(final, (scenario=sc.name, seed, variant=v.label,
            init_mu=r.bias[1], init_logscale=r.bias[2],
            best_iter=r.best, iters_to_5pct,
            rmse_mu_best=r.at_best.rmse_mu, spearman_best=r.at_best.spearman,
            rmse_mu_end=r.path[end].rmse_mu, spearman_end=r.path[end].spearman))
        for p in r.path
            push!(paths, (scenario=sc.name, seed, variant=v.label, p...))
        end
    end
    @info "done" sc.name seed
end

###############################
# Summary over seeds
###############################
cols = [:best_iter, :iters_to_5pct, :rmse_mu_best, :spearman_best, :rmse_mu_end, :spearman_end]
summary = combine(groupby(final, [:scenario, :variant]), cols .=> mean .=> cols)
for sc in scenarios
    println("\n== $(sc.name) ==")
    show(stdout, filter(:scenario => ==(sc.name), summary)[:, Not(:scenario)]; allrows=true, allcols=true)
end

###############################
# Plot: mean path over seeds, rmse_mu and spearman vs ntrees (log x)
###############################
colors = Dict(
    "gaussian_mle" => "black",
    "student mean/sd" => "red",
    "student mean/mad" => "blue",
    "student median/sd" => "purple",
    "student median/mad" => "green",
)
mpath = combine(groupby(paths, [:scenario, :variant, :ntrees]), [:rmse_mu, :spearman] .=> mean .=> [:rmse_mu, :spearman])

f = Figure(size=(1200, 600))
for (j, sc) in enumerate(scenarios)
    ax1 = Axis(f[1, j]; title="$(sc.name)", xscale=log10, ylabel=j == 1 ? "rmse vs true μ" : "")
    ax2 = Axis(f[2, j]; xscale=log10, xlabel="ntrees", ylabel=j == 1 ? "spearman(μ̂, y)" : "")
    for v in variants
        s = sort(filter(r -> r.scenario == sc.name && r.variant == v.label, mpath), :ntrees)
        lines!(ax1, s.ntrees, s.rmse_mu; color=colors[v.label], label=v.label)
        lines!(ax2, s.ntrees, s.spearman; color=colors[v.label])
    end
    j == 1 && Legend(f[3, 1:length(scenarios)], ax1; orientation=:horizontal, tellwidth=false)
end
save(joinpath(assets, "student-init-convergence-$tree_type-$_device.png"), f)

"""
Median/MAD gives Student a starting point that sits where the bulk of the data is and a correct sense of what an outlier is. 
On clean data that hardly matters.
"""