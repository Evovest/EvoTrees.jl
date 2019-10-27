using StaticArrays
using Base.Threads: @threads
using StatsBase: sample
using BenchmarkTools
using EvoTrees
using EvoTrees: find_split_static!, pred_leaf, update_hist!, find_split_narrow!

x1 = BitSet([1,2,3,4,5,8,9])
x2 = BitSet([2,5,8])
x3 = BitSet([1,3])
x4 = BitSet([4,9])
bags = [[x2, x3, x4]]
intersect(x1, x2)

x1 = [1,2,3,4,5,8,9]
x2 = [2,5,8]
x3 = [1,3]
x4 = [4,9]
bags = [[x2, x3, x4]]
intersect(x1, x2)

nrows = 100_000
ncols = 100
nbins = 32

id1_int = sample(1:nrows, Int(nrows/2), replace=false, ordered=true)
id2_int = sample(1:nrows, Int(nrows/4), replace=false, ordered=false)

id1_bit = BitSet(sample(1:nrows, Int(nrows/2), replace=false, ordered=true));
id2_bit = BitSet(sample(1:nrows, Int(nrows/4), replace=false, ordered=false));

hist = zeros(32)
x_bin = sample(UInt8.(1:nbins), nrows, replace=true, ordered=false)
value = rand(nrows)

function hist_sum1(x::AbstractVector{S}, hist::AbstractVector{T}, set::I, value::AbstractVector{T}) where {S,T,I}
    hist .*= 0
    for i in set
        hist[x[i]] += value[i]
    end
    return
end

function hist_sum2(x, hist, set, value)
    hist .= 0
    for i in set
        hist[x[i]] += value[i]
    end
    return
end
@btime hist_sum1($x_bin, $hist, $id1_int, $value)
@btime hist_sum1($x_bin, $hist, $id1_bit, $value)

@btime hist_sum2($x_bin, $hist, $id1_int, $value)
@btime hist_sum2($x_bin, $hist, $id1_bit, $value)

function zeroise1(hist)
    hist .*= 0.0
end
function zeroise2(hist)
    hist .= 0.0
end
function zeroise3(hist)
    hist .= zero(eltype(hist))
end

hist_v = zeros(32)
hist_s = zeros(SVector{1, Float64}, 32)
@btime zeroise1($hist_v)
@btime zeroise2($hist_v)
@btime zeroise3($hist_v)
@btime zeroise1($hist_s)
@btime zeroise2($hist_s)

function zeroise3(hist)
    hist .= [zero(eltype(hist))]
end

hist_s = zeros(SVector{1, Float64}, 32)
@btime zeroise1($hist_s)
@btime zeroise3($hist_s)

function inter(x1, x2)
    res = intersect(x1, x2)
    return res
end
@btime inter($id1_int, $id2_int)
@btime inter($id1_bit, $id2_bit)

@time inter(id1_int, id2_int)
@time inter(id1_bit, id2_bit)
@time inter(id1_int, id2_bit)

function pushtest(src, cond, child)
    for i in src
        if in(i, cond)
            push!(child, i)
        end
    end
end

function conv(x::Vector{Int})
    BitSet(x)
end

function conv(x::BitSet)
    Int.(x)
end

src = sample(1:nrows, nrows, replace=false, ordered=false)
cond = sample(1:nrows, 50_000, replace=false, ordered=false)
cond = BitSet(cond);
child = Vector{Int}()
@time pushtest(src, cond, child)

@time src_bit = conv(src);
@time src_int = conv(src_bit)


x1 = rand(100_000, 100)

function sum(x::Matrix{T}, hist::Matrix{T})

    res = intersect(x1, x2)
    return res
end


intersect.(Ref(x1), bags[1])

@inline function update_bags_intersect(bags, set)
    # new_bags = Vector{Vector{BitSet}}(undef, 10)
    new_bags = [Vector{BitSet}() for i in 1:length(bags)]
    @threads for feat in 1:length(bags)
        new_bags[feat] = intersect.(bags[feat], Ref(set))
        # new_bags[feat] = intersect.(Ref(set), bags[feat])
    end
    return new_bags
end

nbins = 64
bin_size = 5_000
𝑖 = 1:nbins*bin_size |> collect
set = BitSet(𝑖);
bags = [Vector{BitSet}() for i in 1:nbins]
for i in 1:length(bags)
    x = sample(𝑖, length(𝑖), replace=false)
    bags[i] = [BitSet(x[bin_size*(i-1)+1:bin_size*i]) for i in 1:nbins]
end

length(bags[1][1])

set = BitSet(sample(𝑖, 10_000, replace=false));
@time new_bags = update_bags_intersect(bags, set);
length(new_bags[2][2])
