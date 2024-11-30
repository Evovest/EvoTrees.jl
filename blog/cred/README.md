# Exploring a credibility-based approach for tree-gain estimation

The motivation for this experiment stems from potential shortfalls in approach used in gradoent-boosted trees to assess the best split potential.

````julia
# compare mse vs cred based metrics
using EvoTrees
using DataFrames
using Distributions
using Statistics: mean, std
using CairoMakie

function get_∑(p::Matrix{T}, y::Vector{T}, params) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    EvoTrees.update_grads!(∇, p, y, params)
    ∑ = dropdims(sum(∇, dims=2), dims=2)
    return ∑
end

function simul_Z(; nobs, loss, spread=1.0, sd=1.0)
    config = EvoTreeRegressor(; loss)
    p = zeros(1, nobs)
    y = randn(nobs)
    _std = length(y) == 1 ? abs(first(y)) : std(y; corrected=false)
    y .= (y .- mean(y)) ./ _std .* sd .- spread
    ∑ = get_∑(p, y, config)
    Z = EvoTrees._get_cred(config, ∑)
    return Z
end

function get_figure(;
    loss,
    sd,
    nobs_list,
    spread_list)

    xticks = string.(nobs_list)
    yticks = string.(spread_list)

    matrix = zeros(length(nobs_list), length(spread_list))

    for (idx, nobs) in enumerate(nobs_list)
        for (idy, spread) in enumerate(spread_list)
            @info "nobs: $nobs | spread: $spread"
            z = simul_Z(; loss, nobs, spread, sd)
            matrix[idx, idy] = z
        end
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; title="$(string(loss)) | sd: $sd", xlabel="nobs", ylabel="spread", xticks=(1:length(xticks), xticks), yticks=(1:length(yticks), yticks))
    heat = heatmap!(ax, matrix)
    Colorbar(fig[2, 1], heat; vertical=false)
    return fig
````

return matrix

````julia
end

sd = 1.0
nobs_list = Int.(10.0 .^ (0:6))
nobs_list[1] = 2
spread_list = [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 100]

get_figure(; loss=:credV1A, sd, nobs_list, spread_list)
get_figure(; loss=:credV2A, sd, nobs_list, spread_list)

get_figure(; loss=:credV1B, sd, nobs_list, spread_list)
get_figure(; loss=:credV2B, sd, nobs_list, spread_list)

f1 = get_figure(; loss=:credV1A, sd, nobs_list, spread_list)
f2 = get_figure(; loss=:credV2A, sd, nobs_list, spread_list)

f1 = get_figure(; loss=:credV1A, sd, nobs_list, spread_list)
f2 = get_figure(; loss=:credV2A, sd, nobs_list, spread_list)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

