using Statistics
using Base.Threads: @threads

# prepare a dataset
X = rand(Int(1.25e6), 100)

function get_edges(X::AbstractMatrix{T}, nbins=250) where {T}
    edges = Vector{Vector{T}}(undef, size(X,2))
    @threads for i in 1:size(X, 2)
    # for i in 1:size(X, 2)
        edges[i] = quantile(view(X, :,i), (1:nbins)/nbins)
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :,i))]
        end
    end
    return edges
end

println("num threads: ", Threads.nthreads())

println("trial 1: ")
edges = get_edges(X, 128);

println("trial 2: ")
edges = get_edges(X, 128);

println("trial 3: ")
edges = get_edges(X, 128);

println("trial 4: ")
edges = get_edges(X, 128);

println("trial 5: ")
edges = get_edges(X, 128);

println("trial 6: ")
edges = get_edges(X, 128);

println("trial 7: ")
edges = get_edges(X, 128);

println("trial 8: ")
edges = get_edges(X, 128);