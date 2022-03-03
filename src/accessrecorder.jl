"""
    accessrecorder(mutable) -> recorder

Return a `recorder` such that index and field access yield a lens to the
corresponding memory location.

TODO: record field access

TODO: record nested field and index access to update nested element (e.g.,
array-of-large-structs) atomically?
"""
function accessrecorder end

accessrecorder(xs::AbstractArray) = LensArray(xs)

const IntIndexLens{N} = Atomix.IndexLens{NTuple{N,Int}}

# TODO: don't subtype `AbstractArray`; it is crazy that `xs[1]` and `xs[1, 1]`
#       where `xs::LensArray{2}` returns a different value.
# TODO: only store `axes(data)`?
struct LensArray{N,Data<:AbstractArray{<:Any,N}} <:
       AbstractArray{N,Union{IntIndexLens{N},IntIndexLens{1}}}
    data::Data
end

Base.size(a::LensArray) = size(a.data)
Base.IndexStyle(::Type{<:LensArray{<:Any,Data}}) where {Data} = Base.IndexStyle(Data)

@propagate_inbounds function Base.getindex(a::LensArray, i::Int)
    @boundscheck checkbounds(a.data, i)
    return IntIndexLens{1}((i,))
end

@propagate_inbounds function Base.setindex(a::LensArray{N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a.data, I...)
    return IntIndexLens{N}(I)
end

@inline Atomix.pointer(a::AbstractArray, lens::IntIndexLens{1}) =
    pointer(a, lens.indices[1])

@inline function Atomix.pointer(a::AbstractArray{N}, lens::IntIndexLens{N}) where {N}
    i = LinearIndices(data)[lens.indices...]
    return pointer(a, i)
end
