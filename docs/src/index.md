# EvoTress.jl

A Julia implementation of boosted trees with CPU and GPU support. Efficient histogram based algorithms with support for multiple loss functions, including various regressions, multi-classification and Gaussian max likelihood. 

See the `examples-API` section to get started using the internal API, or `examples-MLJ` to use within the MLJ framework.

Complete details about hyper-parameters are found in the `Models` section.

[R binding available](https://github.com/Evovest/EvoTrees).

## Installation

Latest:

```julia-repl
julia> Pkg.add("https://github.com/Evovest/EvoTrees.jl")
```

From General Registry:

```julia-repl
julia> Pkg.add("EvoTrees")
```
