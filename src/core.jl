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

# Native atomic operations for non-Complex types
_atomic_load(ptr::Ptr{T}, order) where {T} = 
    UnsafeAtomics.load(ptr, order)

_atomic_store!(ptr::Ptr{T}, val::T, order) where {T} = 
    UnsafeAtomics.store!(ptr, val, order)

_atomic_cas!(ptr::Ptr{T}, expected::T, desired::T, success_order, failure_order) where {T} = 
    UnsafeAtomics.cas!(ptr, expected, desired, success_order, failure_order)

_atomic_modify!(ptr::Ptr{T}, op::OP, x::T, ord) where {T,OP} =
    UnsafeAtomics.modify!(ptr, op, x, ord)

# Complex atomic operations via integer reinterpretation
# Note: Can't use Core.LLVMPtr here since Ptr is not a subtype of Core.LLVMPtr
# So we need separate implementations for CPU (Ptr) and GPU (Core.LLVMPtr)

@inline function _atomic_cas!(ptr::Ptr{Complex{T}}, expected::Complex{T}, desired::Complex{T}, success_order, failure_order) where {T<:Union{Float32,Float64}}
    IntType = _int_type_for_complex(Complex{T})
    int_ptr = reinterpret(Ptr{IntType}, ptr)
    expected_i = reinterpret(IntType, expected)
    desired_i = reinterpret(IntType, desired)
    result = UnsafeAtomics.cas!(int_ptr, expected_i, desired_i, success_order, failure_order)
    return (old = reinterpret(Complex{T}, result.old), success = result.success)
end

@inline function _atomic_load(ptr::Ptr{Complex{T}}, order) where {T<:Union{Float32,Float64}}
    IntType = _int_type_for_complex(Complex{T})
    int_ptr = reinterpret(Ptr{IntType}, ptr)
    result = UnsafeAtomics.load(int_ptr, order)
    return reinterpret(Complex{T}, result)
end

@inline function _atomic_store!(ptr::Ptr{Complex{T}}, val::Complex{T}, order) where {T<:Union{Float32,Float64}}
    IntType = _int_type_for_complex(Complex{T})
    int_ptr = reinterpret(Ptr{IntType}, ptr)
    val_i = reinterpret(IntType, val)
    UnsafeAtomics.store!(int_ptr, val_i, order)
end

# CAS loop fallback for Complex modify! (following CUDA.jl pattern)
@inline function _atomic_modify!(ptr::Ptr{Complex{T}}, op::OP, x::Complex{T}, ord) where {T<:Union{Float32,Float64},OP}
    old = _atomic_load(ptr, ord)
    while true
        new = op(old, x)
        result = _atomic_cas!(ptr, old, new, ord, ord)
        result.success && return (old => new)
        old = result.old
    end
end

Atomix.asstorable(ref, v) = convert(eltype(ref), v)
