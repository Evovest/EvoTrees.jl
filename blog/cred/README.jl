# # Exploring a credibility-based approach for tree-gain estimation

include(joinpath(@__DIR__, "utils.jl")); #hide

#=
> The motivation for this experiment stems from potential benefits in alternatives to gradient-based gains to identify the best split candidate. 
=#

#=
The figures below illustrate the behavior of vanilla gradient-based approach using a mean-squarred error (MSE) loss.
The 2 colors represent the observations belonging to the left and right children.
Key observations:
- **the gain is invariant to the volatility**: the top vs bottom figures differs only by the std dev of the observations. 
    The associated gain is identical, which is aligned with the gradient-based approach to gain: the gain matches the reduction in the MSE, which is identical regardless of the dispersion. It's strictly driven by their mean.
- **the gain scales linearly with the number of observations**: the right vs left figures contrasts different number of observations (100 vs 10k), and show that gain is directly proportional.
- **the gain scales quadratically with the spread**: moving from a spread of 1.0 to 0.1 between the 2nd and 3rd row results in a drop by 100x of the gain: from 50.0 to 0.5.
=#

loss = :mse#hide
f = get_dist_figure(; loss, nobs=100, spread=1.0, sd=1.0)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-1A.png"), f)#hide
f = get_dist_figure(; loss, nobs=10_000, spread=1.0, sd=1.0)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-1B.png"), f)#hide
f = get_dist_figure(; loss, nobs=100, spread=1.0, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-2A.png"), f)#hide
f = get_dist_figure(; loss, nobs=10_000, spread=1.0, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-2B.png"), f);#hide
f = get_dist_figure(; loss, nobs=100, spread=0.1, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-3A.png"), f)#hide
f = get_dist_figure(; loss, nobs=10_000, spread=0.1, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-mse-3B.png"), f);#hide

#=
| ![](assets/dist-mse-1A.png) | ![](assets/dist-mse-1B.png) |
|:----------------------:|:----------------------:|
| ![](assets/dist-mse-2A.png) | ![](assets/dist-mse-2B.png) |
| ![](assets/dist-mse-3A.png) | ![](assets/dist-mse-3B.png) |
=#

#=
## Credibility-based gains
=#

#=
The idea is for the *gain* to reflect varying uncertainty levels for observations associated to each of the tree-split candidates. 
For tree-split candidates with an identical spread, the intuition is that candidates with a lower volatility, all other things being equal, should be preferred.
The original inspiration comes from credibility theory, a foundational notion in actuarial science with direct connexion mixed effect models and bayesian theory. 
Key concept is that the credibility associated with a set of observations is driven by the relative effect of 2 components:
 - **Variance of the Hypothetical Means (VHM)**: if large differences between candidates means are expected, a greater credibility is assigned.
 - **Expected Value of the Process Variance (EVPV)**: if the data generation process of a given candidate has a large volatility, a smaller credibility is assigned.
The Buhlmann credibility states that the optimal linear posterior estimator of a group mean is: 
 - `Z * X̄ + (1 - Z) * μ`, where `X̄` is the group mean and `μ` the population mean.
=#

#=
This approach results in a shift of perspective in how the gain is derived. 
Classical gradient based is about deriving a second-order approximation of the loss curve for a tre-split candidate.
The gain corresponds to the reduction in this approximated loss by taking the prediciton that minimises the quadratic loss curve. 
The credibility-based takes a loss function agnostic approach, and view the gain as the total absolute change in the credibility-adjusted predicted value.
Example, if a child has a mean residual of *2.0*, credibility of 0.5 and 100 observations, the resulting gain is: `2.0 * 0.5 * 100 = 100.0`, where `2.0 * 0.5` corresponds to the credibility adjusted prediction.
VHM is estimated as the square of the mean of the spread between observed values and predictions: 
 - `VHM = E[X] = mean(y - p)`
EVPV is estimated as the variance of the observations. This value can be derived from the aggregation of the first and second moment of the individual observations: 
 - `EVPV = E[(x - μ)²] = E[X²] - E²[X]`
=#

#=
### Credibility-based gains - credV1A
Same as for the previous the gradient-based MSE error, the gain grows linearly with the number of observations, all other things being equal.
However, a smaller volatility results in an increased gain, as shown in 2nd vs 1st row. 
=#

loss = :credV1A#hide
f = get_dist_figure(; loss, nobs=100, spread=1.0, sd=1.0)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-1A.png"), f);#hide
f = get_dist_figure(; loss, nobs=10_000, spread=1.0, sd=1.0)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-1B.png"), f);#hide
f = get_dist_figure(; loss, nobs=100, spread=1.0, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-2A.png"), f);#hide
f = get_dist_figure(; loss, nobs=10_000, spread=1.0, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-2B.png"), f);#hide
f = get_dist_figure(; loss, nobs=100, spread=0.1, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-3A.png"), f);#hide
f = get_dist_figure(; loss, nobs=10_000, spread=0.1, sd=0.1)#hide
save(joinpath(@__DIR__, "assets", "dist-credV1A-3B.png"), f);#hide

#=
| ![](assets/dist-credV1A-1A.png) | ![](assets/dist-credV1A-1B.png) |
|:----------------------:|:----------------------:|
| ![](assets/dist-credV1A-2A.png) | ![](assets/dist-credV1A-2B.png) |
| ![](assets/dist-credV1A-3A.png) | ![](assets/dist-credV1A-3B.png) |
=#

#=
## Credibility figures
Four credibility variations are being tested:
 - **credV1A**: `VHM / VHM + EVPV`
 - **credV1B**: `VHM / VHM + EVPV / N`
 - **credV2A**: `sqrt(VHM) / sqrt(VHM) + sqrt(EVPV)`
 - **credV2B**: `sqrt(VHM) / sqrt(VHM) + sqrt(EVPV / N)`
The figures below present the credibility factor associated with different spreads and number observations
=#

## simulation grid
sd = 1.0
nobs_list = Int.(10.0 .^ (0:6))
nobs_list[1] = 2
spread_list = [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 100]
f = get_cred_figure(; loss=:credV1A, sd, nobs_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmap-credV1A.png"), f);#hide
f = get_cred_figure(; loss=:credV1B, sd, nobs_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmap-credV1B.png"), f);#hide
f = get_cred_figure(; loss=:credV2A, sd, nobs_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmap-credV2A.png"), f);#hide
f = get_cred_figure(; loss=:credV2B, sd, nobs_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmap-credV2B.png"), f);#hide

#=
| ![](assets/heatmap-credV1A.png) | ![](assets/heatmap-credV1B.png) |
|:----------------------:|:----------------------:|
| ![](assets/heatmap-credV2A.png) | ![](assets/heatmap-credV2B.png) |
=#

## simulation grid
nobs = 100
sd_list = [0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
spread_list = [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 100]
f = get_cred_figureB(; loss=:credV1A, nobs, sd_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmapB-credV1A.png"), f);#hide
f = get_cred_figureB(; loss=:credV1B, nobs, sd_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmapB-credV1B.png"), f);#hide
f = get_cred_figureB(; loss=:credV2A, nobs, sd_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmapB-credV2A.png"), f);#hide
f = get_cred_figureB(; loss=:credV2B, nobs, sd_list, spread_list)#hide
save(joinpath(@__DIR__, "assets", "heatmapB-credV2B.png"), f);#hide

#=
| ![](assets/heatmapB-credV1A.png) | ![](assets/heatmapB-credV1B.png) |
|:----------------------:|:----------------------:|
| ![](assets/heatmapB-credV2A.png) | ![](assets/heatmapB-credV2B.png) |
=#


#=
## Results
From [MLBenchmarks.jl](https://github.com/Evovest/MLBenchmarks.jl).
| **model** | **metric** | **ref** | **credV1A** | **credV2A** | **credV1B** | **credV2B** |
|:---------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|:-----------:|
| boston    | mse        | 6.3     | 6.18        | 6.01        | 6.47        | 6.18        |
| boston    | gini       | 0.945   | 0.948       | 0.953       | 0.949       | 0.944       |
| year      | mse        | 74.9    | 78.3        | 74.3        | 77.6        | 76.9        |
| year      | gini       | 0.662   | 0.645       | 0.667       | 0.653       | 0.651       |
| msrank    | mse        | 0.55    | 0.56        | 0.55        | 0.552       | 0.552       |
| msrank    | ndcg       | 0.511   | 0.499       | 0.509       | 0.509       | 0.509       |
| yahoo     | mse        | 0.565   | 0.587       | 0.569       | missing     | missing     |
| yahoo     | ndcg       | 0.795   | 0.787       | 0.793       | missing     | missing     |
=#
