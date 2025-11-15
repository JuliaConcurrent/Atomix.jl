# TODO: respect ordering
module AtomixCUDAExt

using Atomix: Atomix, IndexableRef
using Atomix.Internal: _int_type_for_complex, _atomic_cas_generic!, _atomic_op_cas_loop!
using CUDA: CUDA, CuDeviceArray

const CuIndexableRef{Indexable<:CuDeviceArray} = IndexableRef{Indexable}

function Atomix.get(ref::CuIndexableRef, order)
    error("not implemented")
end

function Atomix.set!(ref::CuIndexableRef, v, order)
    error("not implemented")
end

@inline function Atomix.replace!(
    ref::CuIndexableRef,
    expected,
    desired,
    success_ordering,
    failure_ordering,
)
    ptr = Atomix.pointer(ref)
    expected = convert(eltype(ref), expected)
    desired = convert(eltype(ref), desired)
    old = _cuda_atomic_cas!(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

# Native CUDA CAS for supported types
@inline function _cuda_atomic_cas!(ptr::Core.LLVMPtr{T,A}, cmp::T, new::T) where {T,A}
    CUDA.atomic_cas!(ptr, cmp, new)
end

# Complex CAS via integer reinterpretation (following CUDA.jl pattern for floats)
# Note: ComplexF64 requires UInt128 which is not supported by CUDA atomic_cas!
# So we only support ComplexF32 which uses UInt64
@inline function _cuda_atomic_cas!(ptr::Core.LLVMPtr{ComplexF32,A}, cmp::ComplexF32, new::ComplexF32) where {A}
    _atomic_cas_generic!(ptr, cmp, new, CUDA.atomic_cas!)
end

@inline function Atomix.modify!(ref::CuIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = _cuda_atomic_modify!(ptr, op, x)
    return old => op(old, x)
end

# Native CUDA atomic operations for supported types
@inline function _cuda_atomic_modify!(ptr::Core.LLVMPtr{T,A}, op::OP, x::T) where {T,A,OP}
    if op === (+)
        CUDA.atomic_add!(ptr, x)
    elseif op === (-)
        CUDA.atomic_sub!(ptr, x)
    elseif op === (&)
        CUDA.atomic_and!(ptr, x)
    elseif op === (|)
        CUDA.atomic_or!(ptr, x)
    elseif op === xor
        CUDA.atomic_xor!(ptr, x)
    elseif op === min
        CUDA.atomic_min!(ptr, x)
    elseif op === max
        CUDA.atomic_max!(ptr, x)
    else
        error("not implemented")
    end
end

# Complex atomic operations via CAS loop (following CUDA.jl pattern)
# Note: ComplexF64 requires UInt128 which is not supported by CUDA atomic_cas!
# So we only support ComplexF32 which uses UInt64
@inline function _cuda_atomic_modify!(ptr::Core.LLVMPtr{ComplexF32,A}, op::OP, x::ComplexF32) where {A,OP}
    _atomic_op_cas_loop!(ptr, op, x, CUDA.atomic_cas!)
end

end  # module AtomixCUDAExt
