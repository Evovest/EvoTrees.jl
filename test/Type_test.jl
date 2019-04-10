struct x1{T<:AbstractFloat, S<:Int}
    allo::T
    bonjour::T
    salut::S
end

function f1(x::B) where {T<:AbstractFloat, S<:Int, B<:Vector{x1{T, S}}}
    a = x[1]
    a.allo + a.bonjour
end

a1 = x1(1.0, 2.0, 3)
@time r1 = f1([a1])
typeof(r1)

a1 = x1{Float16, Int}(1.0, 2.0, 3)
@time r1 = f1([a1])
typeof(r1)
