module AtomixMetalExt

using Atomix: Atomix, IndexableRef, right
using Metal: Metal, MtlDeviceArray
using Core: LLVMPtr

const MtlIndexableRef{Indexable<:MtlDeviceArray} = IndexableRef{Indexable}

# Everything goes through Metal.jl's `atomic_*_explicit` intrinsics, using their default
# (relaxed) memory order: the ordering Atomix is asked for is ignored.

const NativeInt = Union{Int32,UInt32}

@inline function Atomix.get(ref::MtlIndexableRef, order)
    ptr = Atomix.pointer(ref)
    return Metal.atomic_load_explicit(ptr)
end

@inline function Atomix.set!(ref::MtlIndexableRef, v, order)
    v = convert(eltype(ref), v)
    ptr = Atomix.pointer(ref)
    Metal.atomic_store_explicit(ptr, v)
    return
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
    old = Metal.atomic_compare_exchange_weak_explicit(ptr, expected, desired)
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(ref::MtlIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = modify_native!(ptr, op, x)
    return old => op(old, x)
end

# operations with a native atomic instruction
for (op, fn) in [(+) => :atomic_fetch_add_explicit, (-) => :atomic_fetch_sub_explicit,
                 (&) => :atomic_fetch_and_explicit, (|) => :atomic_fetch_or_explicit,
                 xor => :atomic_fetch_xor_explicit, min => :atomic_fetch_min_explicit,
                 max => :atomic_fetch_max_explicit]
    @eval @inline modify_native!(ptr::LLVMPtr{<:NativeInt}, ::typeof($op), x) =
        Metal.$fn(ptr, x)
end
# Float32 add/sub is native on device memory only; threadgroup memory takes the
# compare-and-swap path below.
@inline modify_native!(ptr::LLVMPtr{Float32,Metal.AS.Device}, ::typeof(+), x) =
    Metal.atomic_fetch_add_explicit(ptr, x)
@inline modify_native!(ptr::LLVMPtr{Float32,Metal.AS.Device}, ::typeof(-), x) =
    Metal.atomic_fetch_sub_explicit(ptr, x)
@inline modify_native!(ptr::LLVMPtr{<:Union{NativeInt,Float32}}, ::typeof(right), x) =
    Metal.atomic_exchange_explicit(ptr, x)

# everything else (float min/max, arbitrary functions): compare-and-swap loop
@inline modify_native!(ptr::LLVMPtr, op, x) = Metal.atomic_fetch_op_explicit(ptr, op, x)

end  # module AtomixMetalExt
