## package loading
using EvoTrees
using EvoTrees: _get_cred, _loss2type_dict, update_grads!, Cred, EvoTypes, GradientRegression
using DataFrames
using Distributions
using Statistics: mean, std
using CairoMakie

function _get_gain(::Type{L}, params::EvoTypes, ∑::AbstractVector{T}) where {L,T}
    ϵ = eps(T)
    lambda = params.lambda
    L2 = params.L2
    ∑[1]^2 / max(ϵ, (∑[2] + lambda * ∑[3] + L2)) / 2
end

function get_∑(p::Matrix{T}, y::Vector{T}, L, config) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    update_grads!(∇, p, y, L, config)
    ∑ = dropdims(sum(∇, dims=2), dims=2)
    return ∑
end

# visible code
function get_simul_metric(metric_name="cred"; nobs, loss, spread=1.0, sd=1.0)
    config = EvoTreeRegressor(; loss, L2=0, lambda=0)
    L = _loss2type_dict[config.loss]
    p = zeros(1, nobs)
    y = randn(nobs)
    _std = length(y) == 1 ? abs(first(y)) : std(y; corrected=false)
    y .= (y .- mean(y)) ./ _std .* sd .- spread
    ∑ = get_∑(p, y, L, config)

    if metric_name == "cred"
        metric = _get_cred(L, config, ∑)
    elseif metric_name == "gain"
        metric = _get_gain(L, config, ∑)
    else
        error("metric_name must be either 'cred' or 'cred_std'")
    end
    return metric
end

function get_data(; loss, nobs, spread=1.0, sd=1.0)
    config = EvoTreeRegressor(; loss, L2=0, lambda=0)
    L = _loss2type_dict[config.loss]

    yL, yR = randn(nobs), randn(nobs)
    yL .= (yL .- mean(yL)) ./ std(yL) .* sd .- spread / 2
    yR .= (yR .- mean(yR)) ./ std(yR) .* sd .+ spread / 2
    yT = vcat(yL, yR)

    pL = zeros(1, nobs)
    pR = zeros(1, nobs)
    pT = zeros(1, 2 * nobs)

    data = Dict()
    data[:yL] = yL
    data[:yR] = yR

    ## gains
    ∑T = get_∑(pT, yT, L, config)
    ∑L = get_∑(pL, yL, L, config)
    ∑R = get_∑(pR, yR, L, config)
    data[:gP] = _get_gain(L, config, ∑T)
    data[:gL] = _get_gain(L, config, ∑L)
    data[:gR] = _get_gain(L, config, ∑R)
    data[:gC] = data[:gL] + data[:gR]

    if L <: Cred
        data[:ZR] = _get_cred(L, config, ∑R)
    else
        data[:ZR] = NaN
    end

    return data
end

function get_dist_figure(; loss, nobs, spread=1.0, sd=1.0)

    data = get_data(; loss, nobs, spread, sd)

    gP = round(data[:gP]; digits=3)
    gC = round(data[:gC]; sigdigits=4)
    gL = round(data[:gL]; sigdigits=4)
    gR = round(data[:gR]; sigdigits=4)
    ZR = round(data[:ZR]; sigdigits=4)

    f = Figure()
    ax1 = Axis(f[1, 1];
        title="nobs=$nobs | spread=$spread | sd=$sd",
        subtitle=
        """
        gain parent=$gP | gain cildren=$gC
        gainL=$gL | gainR=$gR | ZR=$ZR
        """
    )
    density!(ax1, data[:yL]; color="#4571a5CC", label="left")
    density!(ax1, data[:yR]; color="#26a671CC", label="right")
    Legend(f[2, 1], ax1, orientation=:horizontal)
    return f
end

function get_dist_figure_juliacon(; loss, nobs, spread=1.0, sd=1.0)

    data = get_data(; loss, nobs, spread, sd)

    gP = round(data[:gP]; digits=3)
    gC = round(data[:gC]; sigdigits=4)
    gL = round(data[:gL]; sigdigits=4)
    gR = round(data[:gR]; sigdigits=4)
    ZR = round(data[:ZR]; sigdigits=4)

    f = Figure()
    ax1 = Axis(f[1, 1];
        title="nobs=$nobs | spread=$spread | sd=$sd",
        subtitle=
        """
        gainL=$gL | gainR=$gR
        """,
        titlesize=18,
        subtitlesize=18,
        xlabelsize=18,
        ylabelsize=18,
        xticklabelsize=18,
        yticklabelsize=18,
    )
    density!(ax1, data[:yL]; color="#4571a5CC", label="left")
    density!(ax1, data[:yR]; color="#26a671CC", label="right")
    Legend(f[2, 1], ax1, orientation=:horizontal, labelsize=18)
    return f
end

function get_cred_figure(;
    metric_name="cred",
    loss,
    sd,
    nobs_list,
    spread_list)

    xticks = string.(nobs_list)
    yticks = string.(spread_list)

    matrix = zeros(length(nobs_list), length(spread_list))

    for (idx, nobs) in enumerate(nobs_list)
        for (idy, spread) in enumerate(spread_list)
            metric = get_simul_metric(metric_name; loss, nobs, spread, sd)
            matrix[idx, idy] = metric
        end
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; title=metric_name, subtitle="$(string(loss)) | sd: $sd", xlabel="nobs", ylabel="spread", xticks=(1:length(xticks), xticks), yticks=(1:length(yticks), yticks))
    heat = heatmap!(ax, matrix)
    Colorbar(fig[2, 1], heat; vertical=false)
    return fig
end

function get_cred_figureB(;
    metric_name="cred",
    loss,
    nobs,
    sd_list,
    spread_list)

    xticks = string.(sd_list)
    yticks = string.(spread_list)

    matrix = zeros(length(sd_list), length(spread_list))

    for (idx, sd) in enumerate(sd_list)
        for (idy, spread) in enumerate(spread_list)
            metric = get_simul_metric(metric_name; loss, nobs, spread, sd)
            matrix[idx, idy] = metric
        end
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; title=metric_name, subtitle="$(string(loss)) | nobs: $nobs", xlabel="sd", ylabel="spread", xticks=(1:length(xticks), xticks), yticks=(1:length(yticks), yticks))
    heat = heatmap!(ax, matrix)
    Colorbar(fig[2, 1], heat; vertical=false)
    return fig
end
