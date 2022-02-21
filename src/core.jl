#=
@inline Base.pointer(A::AtomicArray) = pointer(A.data)
@inline Base.pointer(A::AtomicArray, i::Integer) = pointer(A.data, i)

@propagate_inbounds function atomic_getindex(A::AtomicArray, i::Integer, ord::Ordering)
    @boundscheck checkbounds(A, i)
    data = A.data
    GC.@preserve data begin
        ptr = pointer(data, i)
        v = UnsafeAtomics.load(ptr, ord)
    end
    return v
end

@propagate_inbounds function atomic_setindex!(A::AtomicArray, i::Integer, v, ord::Ordering)
    @boundscheck checkbounds(A, i)
    v = convert(eltype(A), v)
    data = A.data
    GC.@preserve data begin
        ptr = pointer(data, i)
        UnsafeAtomics.store!(ptr, v, ord)
    end
end

@propagate_inbounds function atomic_replaceindex!(
    A::AtomicArray,
    expected,
    desired,
    i::Integer,
    success_ordering::Ordering,
    failure_ordering::Ordering,
)
    @boundscheck checkbounds(A, i)
    expected = convert(eltype(A), expected)
    desired = convert(eltype(A), desired)
    data = A.data
    GC.@preserve data begin
        ptr = pointer(data, i)
        result =
            UnsafeAtomics.cas!(ptr, expected, desired, success_ordering, failure_ordering)
    end
    return result
end

@propagate_inbounds function atomic_modifyindex!(
    A::AtomicArray,
    i::Integer,
    v,
    ord::Ordering,
)
    @boundscheck checkbounds(A, i)
    v = convert(eltype(A), v)
    data = A.data
    GC.@preserve data begin
        ptr = pointer(data, i)
        UnsafeAtomics.store!(ptr, v, ord)
    end
end

@propagate_inbounds function atomic_swapindex!(A::AtomicArray, ord::Ordering, v, i::Integer)
    @boundscheck checkbounds(A, i)
    v = convert(eltype(A), v)
    data = A.data
    GC.@preserve data begin
        ptr = pointer(data, i)
        UnsafeAtomics.store!(ptr, v, ord)
    end
end
=#

@inline AtomicArrays.asref(A) = AtomicRefArray(A)

Base.size(A::AtomicRefArray) = size(A.data)
Base.IndexStyle(::Type{<:AtomicRefArray{<:Any,<:Any,Data}}) where {Data} = Base.IndexStyle(Data)

@propagate_inbounds function Base.getindex(A::AtomicRefArray, i::Int)
    @boundscheck checkbounds(A, i)
    data = A.data
    return AtomicRef(Val{eltype(data)}(), pointer(data, i), data)
end

@propagate_inbounds function Base.getindex(A::AtomicRefArray{<:Any,N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(A, I...)
    data = A.data
    i = LinearIndices(data)[I...]
    return AtomicRef(Val{eltype(data)}(), pointer(data, i), data)
end

@inline function UnsafeAtomics.load(ref::AtomicRef, ord::Ordering)
    ptr = ref.ptr
    data = ref.data
    GC.@preserve data begin
        UnsafeAtomics.load(ptr, ord)
    end
end

@inline function UnsafeAtomics.store!(ref::AtomicRef{T}, v, ord::Ordering) where {T}
    v = convert(T, v)
    ptr = ref.ptr
    data = ref.data
    GC.@preserve data begin
        UnsafeAtomics.store!(ptr, v, ord)
    end
end

@inline function UnsafeAtomics.cas!(
    ref::AtomicRef{T},
    expected,
    desired,
    success_ordering::Ordering,
    failure_ordering::Ordering,
) where {T}
    expected = convert(T, expected)
    desired = convert(T, desired)
    ptr = ref.ptr
    data = ref.data
    GC.@preserve data begin
        UnsafeAtomics.cas!(ptr, expected, desired, success_ordering, failure_ordering)
    end
end

@inline function UnsafeAtomics.modify!(ref::AtomicRef{T}, op::OP, x, ord) where {T,OP}
    x = convert(T, x)
    ptr = ref.ptr
    data = ref.data
    GC.@preserve data begin
        UnsafeAtomics.modify!(ptr, op, x, ord)
    end
end

@inline Base.getindex(ref::AtomicRef) = UnsafeAtomics.load(ref, monotonic)
