module AtomixCUDACoreExt

using Atomix: Atomix, IndexableRef, right
using CUDACore: CUDACore, CuDeviceArray
using Core: LLVMPtr

const CuIndexableRef{Indexable<:CuDeviceArray} = IndexableRef{Indexable}

# `Atomix.get` and `Atomix.set!` use Atomix's generic implementation: an LLVM atomic
# load/store on the device pointer (via UnsafeAtomicsLLVM), which honors the ordering.
#
# `Atomix.replace!` and `Atomix.modify!` go through CUDA.jl's intrinsics. These have no
# ordering parameter, so the requested ordering is ignored: every operation is performed
# with CUDA's default (relaxed, device-scope) semantics.

const NativeInt = Union{Int32,Int64,UInt32,UInt64}
const NativeFloat = Union{Float32,Float64}

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
    old = CUDACore.atomic_cas!(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(ref::CuIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = modify_native!(ptr, op, x)
    return old => op(old, x)
end

# operations with a native atomic instruction
for (op, fn) in [(+) => :atomic_add!, (-) => :atomic_sub!, (&) => :atomic_and!,
                 (|) => :atomic_or!, xor => :atomic_xor!, min => :atomic_min!,
                 max => :atomic_max!]
    @eval @inline modify_native!(ptr::LLVMPtr{<:NativeInt}, ::typeof($op), x) =
        CUDACore.$fn(ptr, x)
end
@inline modify_native!(ptr::LLVMPtr{Float32}, ::typeof(+), x) = CUDACore.atomic_add!(ptr, x)
@inline modify_native!(ptr::LLVMPtr{Float32}, ::typeof(-), x) = CUDACore.atomic_sub!(ptr, x)
# Float64 atomic add needs compute capability 6.0; use compare-and-swap below that.
@inline function modify_native!(ptr::LLVMPtr{Float64}, ::typeof(+), x)
    if CUDACore.compute_capability().major >= 6
        CUDACore.atomic_add!(ptr, x)
    else
        modify_cas!(ptr, +, x)
    end
end
@inline modify_native!(ptr::LLVMPtr{Float64}, ::typeof(-), x) = modify_native!(ptr, +, -x)

# swap: exchange floats through their integer representation
@inline modify_native!(ptr::LLVMPtr{<:NativeInt}, ::typeof(right), x) =
    CUDACore.atomic_xchg!(ptr, x)
for (T, I) in [Float32 => UInt32, Float64 => UInt64]
    @eval @inline function modify_native!(ptr::LLVMPtr{$T,A}, ::typeof(right), x) where {A}
        old = CUDACore.atomic_xchg!(reinterpret(LLVMPtr{$I,A}, ptr), reinterpret($I, x))
        return reinterpret($T, old)
    end
end

# everything else (float min/max, arbitrary functions): compare-and-swap loop
@inline modify_native!(ptr::LLVMPtr, op, x) = modify_cas!(ptr, op, x)

@inline function modify_cas!(ptr::LLVMPtr{T}, op, x) where {T}
    old = Base.unsafe_load(ptr)
    while true
        new = convert(T, op(old, x))
        seen = CUDACore.atomic_cas!(ptr, old, new)
        # bitwise comparison: `==` would spin forever on NaN
        seen === old && return old
        old = seen
    end
end

end  # module AtomixCUDACoreExt
