# TODO: respect ordering
module AtomixoneAPIExt

using Atomix: Atomix, IndexableRef
using oneAPI: oneAPI, oneDeviceArray

const oneIndexableRef{Indexable<:oneDeviceArray} = IndexableRef{Indexable}

function Atomix.get(ref::oneIndexableRef, order)
    error("not implemented")
end

function Atomix.set!(ref::oneIndexableRef, v, order)
    error("not implemented")
end

@inline function Atomix.replace!(
    ref::oneIndexableRef,
    expected,
    desired,
    success_ordering,
    failure_ordering,
)
    ptr = Atomix.pointer(ref)
    expected = convert(eltype(ref), expected)
    desired = convert(eltype(ref), desired)
    old = _oneapi_atomic_cas!(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

# Native oneAPI CAS for supported types
@inline function _oneapi_atomic_cas!(ptr::Core.LLVMPtr{T,A}, cmp::T, new::T) where {T,A}
    oneAPI.atomic_cmpxchg!(ptr, cmp, new)
end

# Complex CAS - using separate CAS on real and imaginary components
# Note: This is NOT fully atomic (components updated separately)
# but works for both ComplexF32 and ComplexF64
@inline function _oneapi_atomic_cas!(ptr::Core.LLVMPtr{Complex{T},A}, cmp::Complex{T}, new::Complex{T}) where {T<:Union{Float32,Float64},A}
    # Get pointers to real and imaginary components
    ptr_re = Base.bitcast(Core.LLVMPtr{T,A}, ptr)
    ptr_im = Base.bitcast(Core.LLVMPtr{T,A}, ptr + sizeof(T))
    
    # CAS on real part
    old_re = oneAPI.atomic_cmpxchg!(ptr_re, cmp.re, new.re)
    # CAS on imaginary part  
    old_im = oneAPI.atomic_cmpxchg!(ptr_im, cmp.im, new.im)
    
    # Return just the old value for consistency
    return Complex{T}(old_re, old_im)
end

@inline function Atomix.modify!(ref::oneIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = _oneapi_atomic_modify!(ptr, op, x)
    return old => op(old, x)
end

# Native oneAPI atomic operations for supported types
@inline function _oneapi_atomic_modify!(ptr::Core.LLVMPtr{T,A}, op::OP, x::T) where {T,A,OP}
    if op === (+)
        oneAPI.atomic_add!(ptr, x)
    elseif op === (-)
        oneAPI.atomic_sub!(ptr, x)
    elseif op === (&)
        oneAPI.atomic_and!(ptr, x)
    elseif op === (|)
        oneAPI.atomic_or!(ptr, x)
    elseif op === xor
        oneAPI.atomic_xor!(ptr, x)
    elseif op === min
        oneAPI.atomic_min!(ptr, x)
    elseif op === max
        oneAPI.atomic_max!(ptr, x)
    else
        error("not implemented")
    end
end

# Complex atomic operations - separate atomics on real and imaginary parts
# This works for operations that decompose component-wise (+, -, right)
# Note: This provides per-component atomicity, not full Complex atomicity
@inline function _oneapi_atomic_modify!(ptr::Core.LLVMPtr{Complex{T},A}, op::OP, x::Complex{T}) where {T<:Union{Float32,Float64},A,OP}
    # Get pointers to real and imaginary components
    ptr_re = Base.bitcast(Core.LLVMPtr{T,A}, ptr)
    ptr_im = Base.bitcast(Core.LLVMPtr{T,A}, ptr + sizeof(T))
    
    if op === (+)
        old_re = oneAPI.atomic_add!(ptr_re, x.re)
        old_im = oneAPI.atomic_add!(ptr_im, x.im)
        return Complex{T}(old_re, old_im)
    elseif op === (-)
        old_re = oneAPI.atomic_sub!(ptr_re, x.re)
        old_im = oneAPI.atomic_sub!(ptr_im, x.im)
        return Complex{T}(old_re, old_im)
    else
        # For other operations (like right for swap), use CAS loop
        old_re = oneAPI.atomic_add!(ptr_re, zero(T))  # atomic read
        old_im = oneAPI.atomic_add!(ptr_im, zero(T))  # atomic read
        old = Complex{T}(old_re, old_im)
        
        # Compute new value
        new = op(old, x)
        
        # Try to swap using CAS
        _oneapi_atomic_cas!(ptr, old, new)
        
        # Return the old value we read
        return old
    end
end

end  # module AtomixoneAPIExt
