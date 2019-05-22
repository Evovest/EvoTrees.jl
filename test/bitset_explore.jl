using StaticArrays
using Base.Threads: @threads
using StatsBase: sample

x1 = BitSet([1,2,3,4,5,8,9])
x2 = BitSet([2,5,8])
x3 = BitSet([1,3])
x4 = BitSet([4,9])

bags = [[x2, x3, x4]]

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
