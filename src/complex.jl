# Shared utilities for Complex atomic operations across all backends
# This file provides common functionality to avoid code repetition

"""
    _int_type_for_complex(::Type{Complex{T}})

Get the integer type that can hold a Complex{T} value via reinterpretation.
ComplexF32 (64 bits) -> UInt64
ComplexF64 (128 bits) -> UInt128
"""
_int_type_for_complex(::Type{Complex{Float32}}) = UInt64
_int_type_for_complex(::Type{Complex{Float64}}) = UInt128

"""
    _pack_complex(c::Complex{Float32})

Pack a ComplexF32 into a UInt64 by packing the two Float32 components.
"""
@inline function _pack_complex(c::Complex{Float32})
    re_bits = Base.bitcast(UInt32, c.re)
    im_bits = Base.bitcast(UInt32, c.im)
    return (UInt64(im_bits) << 32) | UInt64(re_bits)
end

"""
    _unpack_complex(::Type{Complex{Float32}}, bits::UInt64)

Unpack a UInt64 into a ComplexF32 by unpacking the two Float32 components.
"""
@inline function _unpack_complex(::Type{Complex{Float32}}, bits::UInt64)
    re_bits = UInt32(bits & 0xffffffff)
    im_bits = UInt32(bits >> 32)
    re = Base.bitcast(Float32, re_bits)
    im = Base.bitcast(Float32, im_bits)
    return Complex{Float32}(re, im)
end

"""
    _pack_complex(c::Complex{Float64})

Pack a ComplexF64 into a UInt128 by packing the two Float64 components.
"""
@inline function _pack_complex(c::Complex{Float64})
    re_bits = Base.bitcast(UInt64, c.re)
    im_bits = Base.bitcast(UInt64, c.im)
    return (UInt128(im_bits) << 64) | UInt128(re_bits)
end

"""
    _unpack_complex(::Type{Complex{Float64}}, bits::UInt128)

Unpack a UInt128 into a ComplexF64 by unpacking the two Float64 components.
"""
@inline function _unpack_complex(::Type{Complex{Float64}}, bits::UInt128)
    re_bits = UInt64(bits & 0xffffffffffffffff)
    im_bits = UInt64(bits >> 64)
    re = Base.bitcast(Float64, re_bits)
    im = Base.bitcast(Float64, im_bits)
    return Complex{Float64}(re, im)
end

"""
    _atomic_cas_generic!(ptr::Core.LLVMPtr{T,A}, cmp::T, new::T, cas_fn) where {T,A}

Generic atomic compare-and-swap implementation that works for any type T by:
1. Converting pointer to integer pointer type  
2. Manually packing Complex values into integers
3. Performing atomic CAS on the integer representation
4. Unpacking result back to Complex type

This follows the same pattern as CUDA.jl for floating-point atomics.
"""
@inline function _atomic_cas_generic!(ptr::Core.LLVMPtr{Complex{T},A}, cmp::Complex{T}, new::Complex{T}, cas_fn) where {T<:Union{Float32,Float64},A}
    IT = _int_type_for_complex(Complex{T})
    ptr_i = Base.bitcast(Core.LLVMPtr{IT,A}, ptr)
    
    # Pack Complex values to integers
    cmp_i = _pack_complex(cmp)
    new_i = _pack_complex(new)
    
    # Perform CAS on integer representation
    old_i = cas_fn(ptr_i, cmp_i, new_i)
    
    # Unpack result back to Complex
    return _unpack_complex(Complex{T}, old_i)
end

"""
    _atomic_op_cas_loop!(ptr::Core.LLVMPtr{Complex{T},A}, op::OP, val::Complex{T}, cas_fn) where {T,A,OP}

Generic atomic operation implementation using a compare-and-swap loop.
This is the fallback for operations that don't have native hardware support.

This follows the same pattern as CUDA.jl's atomic_op! function.
For Complex types, we need to do an atomic load first via CAS.
"""
@inline function _atomic_op_cas_loop!(ptr::Core.LLVMPtr{Complex{T},A}, op::OP, val::Complex{T}, cas_fn) where {T<:Union{Float32,Float64},A,OP}
    # Atomic load via unsafe_load (the CAS loop below provides atomicity)
    IT = _int_type_for_complex(Complex{T})
    ptr_i = Base.bitcast(Core.LLVMPtr{IT,A}, ptr)
    old_i = Base.unsafe_load(ptr_i)
    old = _unpack_complex(Complex{T}, old_i)
    
    while true
        cmp = old
        new = op(old, val)
        old = _atomic_cas_generic!(ptr, cmp, new, cas_fn)
        isequal(old, cmp) && return old
    end
end
