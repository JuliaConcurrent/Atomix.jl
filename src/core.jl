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

# Complex atomic operations via separate operations on real and imaginary parts
# This provides per-component atomicity (not full Complex atomicity)

@inline function _atomic_load(ptr::Ptr{Complex{T}}, order) where {T<:Union{Float32,Float64}}
    ptr_re = reinterpret(Ptr{T}, ptr)
    ptr_im = reinterpret(Ptr{T}, ptr + sizeof(T))
    re = UnsafeAtomics.load(ptr_re, order)
    im = UnsafeAtomics.load(ptr_im, order)
    return Complex{T}(re, im)
end

@inline function _atomic_store!(ptr::Ptr{Complex{T}}, val::Complex{T}, order) where {T<:Union{Float32,Float64}}
    ptr_re = reinterpret(Ptr{T}, ptr)
    ptr_im = reinterpret(Ptr{T}, ptr + sizeof(T))
    UnsafeAtomics.store!(ptr_re, val.re, order)
    UnsafeAtomics.store!(ptr_im, val.im, order)
end

@inline function _atomic_cas!(ptr::Ptr{Complex{T}}, expected::Complex{T}, desired::Complex{T}, success_order, failure_order) where {T<:Union{Float32,Float64}}
    ptr_re = reinterpret(Ptr{T}, ptr)
    ptr_im = reinterpret(Ptr{T}, ptr + sizeof(T))
    
    # CAS on real part
    result_re = UnsafeAtomics.cas!(ptr_re, expected.re, desired.re, success_order, failure_order)
    # CAS on imaginary part
    result_im = UnsafeAtomics.cas!(ptr_im, expected.im, desired.im, success_order, failure_order)
    
    # Both must succeed for overall success
    success = result_re.success && result_im.success
    return (old = Complex{T}(result_re.old, result_im.old), success = success)
end

@inline function _atomic_modify!(ptr::Ptr{Complex{T}}, op::OP, x::Complex{T}, ord) where {T<:Union{Float32,Float64},OP}
    ptr_re = reinterpret(Ptr{T}, ptr)
    ptr_im = reinterpret(Ptr{T}, ptr + sizeof(T))
    
    # Most operations can be decomposed component-wise for Complex numbers
    # This provides per-component atomicity, not full Complex-level atomicity
    result_re = UnsafeAtomics.modify!(ptr_re, op, x.re, ord)
    result_im = UnsafeAtomics.modify!(ptr_im, op, x.im, ord)
    old = Complex{T}(first(result_re), first(result_im))
    new = Complex{T}(last(result_re), last(result_im))
    return old => new
end

Atomix.asstorable(ref, v) = convert(eltype(ref), v)
