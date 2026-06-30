module AtomixMetalExt

using Atomix: Atomix, IndexableRef
using Metal: Metal, MtlDeviceArray
using UnsafeAtomics: UnsafeAtomics

const MtlIndexableRef{Indexable<:MtlDeviceArray} = IndexableRef{Indexable}

@inline metal_memory_order(::typeof(UnsafeAtomics.unordered)) = Val(Metal.memory_order_relaxed)
@inline metal_memory_order(::typeof(UnsafeAtomics.monotonic)) = Val(Metal.memory_order_relaxed)
@inline metal_memory_order(::typeof(UnsafeAtomics.acquire)) = Val(Metal.memory_order_acquire)
@inline metal_memory_order(::typeof(UnsafeAtomics.release)) = Val(Metal.memory_order_release)
@inline metal_memory_order(::typeof(UnsafeAtomics.acq_rel)) = Val(Metal.memory_order_acq_rel)
@inline metal_memory_order(::typeof(UnsafeAtomics.seq_cst)) = Val(Metal.memory_order_seq_cst)

@inline function Atomix.get(ref::MtlIndexableRef, order)
    ptr = Atomix.pointer(ref)
    return Metal.atomic_load_explicit(ptr, metal_memory_order(order))
end

@inline function Atomix.set!(ref::MtlIndexableRef, v, order)
    v = convert(eltype(ref), v)
    ptr = Atomix.pointer(ref)
    return Metal.atomic_store_explicit(ptr, v, metal_memory_order(order))
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
    begin
        old = Metal.atomic_compare_exchange_weak_explicit(
            ptr, expected, desired,
            metal_memory_order(success_ordering),
            metal_memory_order(failure_ordering),
        )
    end
    return (; old = old, success = old === expected)
end


# CAS is needed for FP ops on ThreadGroup memory
@inline function Atomix.modify!(
    ref::IndexableRef{<:MtlDeviceArray{<:AbstractFloat, <:Any, Metal.AS.ThreadGroup}},
    op::OP,
    x,
    order,
) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    old = Metal.atomic_fetch_op_explicit(ptr, op, x, metal_memory_order(order))
    return old => op(old, x)
end

@inline function Atomix.modify!(ref::MtlIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    metal_order = metal_memory_order(order)
    begin
        old = if op === (+)
            Metal.atomic_fetch_add_explicit(ptr, x, metal_order)
        elseif op === (-)
            Metal.atomic_fetch_sub_explicit(ptr, x, metal_order)
        elseif op === (&)
            Metal.atomic_fetch_and_explicit(ptr, x, metal_order)
        elseif op === (|)
            Metal.atomic_fetch_or_explicit(ptr, x, metal_order)
        elseif op === xor
            Metal.atomic_fetch_xor_explicit(ptr, x, metal_order)
        elseif op === min
            Metal.atomic_fetch_min_explicit(ptr, x, metal_order)
        elseif op === max
            Metal.atomic_fetch_max_explicit(ptr, x, metal_order)
        else
            error("not implemented")
        end
    end
    return old => op(old, x)
end

end  # module AtomixMetalExt
