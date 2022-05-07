"""
    accessrecorder(mutable) -> recorder

Return a `recorder` such that index and field access yield a ref to the corresponding memory
location.

TODO: record field access

TODO: record nested field and index access to update nested element (e.g.,
array-of-large-structs) atomically?
"""
function accessrecorder end

accessrecorder(xs::AbstractArray) = ReferenceableArray(xs)

Base.eltype(::Type{<:IndexableRef{Indexable,<:Any}}) where {Indexable} = eltype(Indexable)

const IntIndexableRef{N,Indexable} = IndexableRef{Indexable,NTuple{N,Int}}

# TODO: Use Referenceables.jl?
# TODO: Don't subtype `AbstractArray`?  It is crazy that `xs[1]` and `xs[1, 1]` where
# `xs::ReferenceableArray{2}` return different values.  Or maybe just define `==` on the
# refs?
struct ReferenceableArray{N,Data<:AbstractArray{<:Any,N}} <:
       AbstractArray{Union{IntIndexableRef{1,Data},IntIndexableRef{N,Data}},N}
    data::Data
end

Base.size(a::ReferenceableArray) = size(a.data)
Base.IndexStyle(::Type{<:ReferenceableArray{<:Any,Data}}) where {Data} =
    Base.IndexStyle(Data)

@propagate_inbounds function Base.getindex(a::ReferenceableArray, i::Int)
    @boundscheck checkbounds(a.data, i)
    return IndexableRef(a.data, (i,))::IntIndexableRef{1}
end

@propagate_inbounds function Base.getindex(
    a::ReferenceableArray{N},
    I::Vararg{Int,N},
) where {N}
    @boundscheck checkbounds(a.data, I...)
    return IndexableRef(a.data, I)::IntIndexableRef{N}
end

@inline Atomix.pointer(ref::IntIndexableRef{1}) = pointer(ref.data, ref.indices[1])

@inline function Atomix.pointer(ref::IntIndexableRef)
    i = LinearIndices(ref.data)[ref.indices...]
    return pointer(ref.data, i)
end

Atomix.gcroot(ref) = ref
Atomix.gcroot(ref::IndexableRef) = ref.data
