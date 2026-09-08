module AtomixoneAPIExt

using Atomix: Atomix, IndexableRef, right
using oneAPI: oneAPI, oneDeviceArray
using Core: LLVMPtr

const oneIndexableRef{Indexable<:oneDeviceArray} = IndexableRef{Indexable}

# `Atomix.get` and `Atomix.set!` use Atomix's generic implementation: an LLVM atomic
# load/store on the device pointer (via UnsafeAtomicsLLVM), which honors the ordering.
#
# `Atomix.replace!` and `Atomix.modify!` go through oneAPI.jl's SPIR-V intrinsics. These
# have no ordering parameter, so the requested ordering is ignored: every operation is
# performed with the intrinsics' default (relaxed, workgroup-scope) semantics.

const NativeInt = Union{Int32,Int64,UInt32,UInt64}
const NativeFloat = Union{Float32,Float64}

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
    old = oneAPI.atomic_cmpxchg!(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(ref::oneIndexableRef, op::OP, x, order) where {OP}
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
        oneAPI.$fn(ptr, x)
end
@inline modify_native!(ptr::LLVMPtr{<:NativeFloat}, ::typeof(+), x) =
    oneAPI.atomic_add!(ptr, x)
@inline modify_native!(ptr::LLVMPtr{<:NativeFloat}, ::typeof(-), x) =
    oneAPI.atomic_sub!(ptr, x)
# floating-point min/max: native or compare-and-swap, depending on the oneAPI.jl version
if hasmethod(oneAPI.atomic_min!, Tuple{LLVMPtr{Float32,1},Float32})
    @inline modify_native!(ptr::LLVMPtr{<:NativeFloat}, ::typeof(min), x) =
        oneAPI.atomic_min!(ptr, x)
    @inline modify_native!(ptr::LLVMPtr{<:NativeFloat}, ::typeof(max), x) =
        oneAPI.atomic_max!(ptr, x)
end
@inline modify_native!(ptr::LLVMPtr{<:Union{NativeInt,NativeFloat}}, ::typeof(right), x) =
    oneAPI.atomic_xchg!(ptr, x)

# everything else: compare-and-swap loop
@inline modify_native!(ptr::LLVMPtr, op, x) = modify_cas!(ptr, op, x)

@inline function modify_cas!(ptr::LLVMPtr{T}, op, x) where {T}
    old = Base.unsafe_load(ptr)
    while true
        new = convert(T, op(old, x))
        seen = oneAPI.atomic_cmpxchg!(ptr, old, new)
        # bitwise comparison: `==` would spin forever on NaN
        seen === old && return old
        old = seen
    end
end

end  # module AtomixoneAPIExt
