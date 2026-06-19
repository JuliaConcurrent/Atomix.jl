# TODO: respect ordering
module AtomixMetalExt

using Atomix: Atomix, IndexableRef
using Metal: Metal, MtlDeviceArray

const MtlIndexableRef{Indexable<:MtlDeviceArray} = IndexableRef{Indexable}

function Atomix.get(ref::MtlIndexableRef, order)
    error("not implemented")
end

function Atomix.set!(ref::MtlIndexableRef, v, order)
    error("not implemented")
end

@inline function Atomix.replace!(
    ref::MtlIndexableRef,
    expected,
    desired,
    success_ordering,
    failure_ordering,
)
    ptr = Atomix.pointer(ref)
    expected = convert(eltype(ref), expected)
    desired = convert(eltype(ref), desired)
    old = _metal_atomic_cas!(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

# Native Metal CAS for supported types
@inline function _metal_atomic_cas!(ptr::Core.LLVMPtr{T,A}, cmp::T, new::T) where {T,A}
    Metal.atomic_compare_exchange_weak_explicit(ptr, cmp, new)
end

# Complex CAS - using separate CAS on real and imaginary components
# Note: This is NOT fully atomic (components updated separately)
# but works for both ComplexF32 and ComplexF64
@inline function _metal_atomic_cas!(ptr::Core.LLVMPtr{Complex{T},A}, cmp::Complex{T}, new::Complex{T}) where {T<:Union{Float32,Float64},A}
    # Get pointers to real and imaginary components
    ptr_re = Base.bitcast(Core.LLVMPtr{T,A}, ptr)
    ptr_im = Base.bitcast(Core.LLVMPtr{T,A}, ptr + sizeof(T))
    
    # CAS on real part
    old_re = Metal.atomic_compare_exchange_weak_explicit(ptr_re, cmp.re, new.re)
    # CAS on imaginary part  
    old_im = Metal.atomic_compare_exchange_weak_explicit(ptr_im, cmp.im, new.im)
    
    # Return just the old value for consistency
    # The caller checks success by comparing old === expected
    return Complex{T}(old_re, old_im)
end


# CAS is needed for FP ops on ThreadGroup memory
@inline function Atomix.modify!(ref::IndexableRef{<:MtlDeviceArray{<:AbstractFloat, <:Any, Metal.AS.ThreadGroup}} , op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = Metal.atomic_fetch_op_explicit(ptr, op, x)
    return old => op(old, x)
end

@inline function Atomix.modify!(ref::MtlIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = _metal_atomic_modify!(ptr, op, x)
    return old => op(old, x)
end

# Native Metal atomic operations for supported types
@inline function _metal_atomic_modify!(ptr::Core.LLVMPtr{T,A}, op::OP, x::T) where {T,A,OP}
    if op === (+)
        Metal.atomic_fetch_add_explicit(ptr, x)
    elseif op === (-)
        Metal.atomic_fetch_sub_explicit(ptr, x)
    elseif op === (&)
        Metal.atomic_fetch_and_explicit(ptr, x)
    elseif op === (|)
        Metal.atomic_fetch_or_explicit(ptr, x)
    elseif op === xor
        Metal.atomic_fetch_xor_explicit(ptr, x)
    elseif op === min
        Metal.atomic_fetch_min_explicit(ptr, x)
    elseif op === max
        Metal.atomic_fetch_max_explicit(ptr, x)
    else
        error("not implemented")
    end
end

# Complex atomic operations - separate atomics on real and imaginary parts
# This works for operations that decompose component-wise (+, -, right)
# Note: This provides per-component atomicity, not full Complex atomicity
@inline function _metal_atomic_modify!(ptr::Core.LLVMPtr{Complex{T},A}, op::OP, x::Complex{T}) where {T<:Union{Float32,Float64},A,OP}
    # Get pointers to real and imaginary components
    ptr_re = Base.bitcast(Core.LLVMPtr{T,A}, ptr)
    ptr_im = Base.bitcast(Core.LLVMPtr{T,A}, ptr + sizeof(T))
    
    if op === (+)
        old_re = Metal.atomic_fetch_add_explicit(ptr_re, x.re)
        old_im = Metal.atomic_fetch_add_explicit(ptr_im, x.im)
        return Complex{T}(old_re, old_im)
    elseif op === (-)
        old_re = Metal.atomic_fetch_sub_explicit(ptr_re, x.re)
        old_im = Metal.atomic_fetch_sub_explicit(ptr_im, x.im)
        return Complex{T}(old_re, old_im)
    else
        # For other operations (like right for swap), use CAS loop
        # Read the old value component by component (not atomic together)
        old_re = Metal.atomic_fetch_add_explicit(ptr_re, zero(T))  # atomic read
        old_im = Metal.atomic_fetch_add_explicit(ptr_im, zero(T))  # atomic read
        old = Complex{T}(old_re, old_im)
        
        # Compute new value
        new = op(old, x)
        
        # Try to swap using CAS
        _metal_atomic_cas!(ptr, old, new)
        
        # Return the old value we read
        return old
    end
end

end  # module AtomixMetalExt
