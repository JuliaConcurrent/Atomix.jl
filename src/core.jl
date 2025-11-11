# Atomic operations with Complex number support via integer reinterpretation

@inline function Atomix.get(ref, order)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        _atomic_load(ptr, order)
    end
end

@inline function Atomix.set!(ref, v, order)
    v = Atomix.asstorable(ref, v)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        _atomic_store!(ptr, v, order)
    end
end

@inline function Atomix.replace!(ref, expected, desired, success_ordering, failure_ordering)
    expected = Atomix.asstorable(ref, expected)
    desired = Atomix.asstorable(ref, desired)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        _atomic_cas!(ptr, expected, desired, success_ordering, failure_ordering)
    end
end

@inline function Atomix.modify!(ref, op::OP, x, ord) where {OP}
    x = Atomix.asstorable(ref, x)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        _atomic_modify!(ptr, op, x, ord)
    end
end

# Generic atomic operations - dispatch on type
_atomic_load(ptr::Ptr{T}, order) where {T} = 
    _with_int_repr(UnsafeAtomics.load, ptr, order)

_atomic_store!(ptr::Ptr{T}, val::T, order) where {T} = 
    _with_int_repr(UnsafeAtomics.store!, ptr, val, order)

_atomic_cas!(ptr::Ptr{T}, expected::T, desired::T, success_order, failure_order) where {T} = 
    _with_int_repr(UnsafeAtomics.cas!, ptr, expected, desired, success_order, failure_order)

# Multiple dispatch for modify! - native atomics for non-Complex types
function _atomic_modify!(ptr::Ptr{T}, op::OP, x::T, ord) where {T,OP}
    UnsafeAtomics.modify!(ptr, op, x, ord)
end

# CAS loop fallback for Complex types (no native atomic modify!)
function _atomic_modify!(ptr::Ptr{Complex{T}}, op::OP, x::Complex{T}, ord) where {T,OP}
    old = _atomic_load(ptr, ord)
    while true
        new = op(old, x)
        result = _atomic_cas!(ptr, old, new, ord, ord)
        result.success && return (old => new)
        old = result.old
    end
end

# Helper: apply atomic operation with integer reinterpretation for Complex types
function _with_int_repr(f, ptr::Ptr{Complex{T}}, args...) where {T}
    IntType = _int_type_for_complex(T)
    int_ptr = reinterpret(Ptr{IntType}, ptr)
    result = f(int_ptr, _to_int.(IntType, args)...)
    return _from_int(Complex{T}, result)
end

_with_int_repr(f, ptr::Ptr{T}, args...) where {T} = f(ptr, args...)

# Integer type mapping for Complex types
_int_type_for_complex(::Type{Float32}) = UInt64
_int_type_for_complex(::Type{Float64}) = UInt128

# Convert to/from integer representation
_to_int(::Type{I}, x::Complex{T}) where {I,T} = reinterpret(I, x)
_to_int(::Type{I}, x) where {I} = x

_from_int(::Type{Complex{T}}, x::Integer) where {T} = reinterpret(Complex{T}, x)
_from_int(::Type{Complex{T}}, result::NamedTuple) where {T} = 
    (old = reinterpret(Complex{T}, result.old), success = result.success)
_from_int(::Type{T}, x) where {T} = x

Atomix.asstorable(ref, v) = convert(eltype(ref), v)
