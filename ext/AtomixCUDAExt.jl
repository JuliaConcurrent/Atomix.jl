# TODO: respect ordering
module AtomixCUDAExt

using Atomix: Atomix, IndexableRef
using Atomix.Internal: UnsafeAtomics
using CUDA: CUDA, CuDeviceArray

const CuIndexableRef{Indexable<:CuDeviceArray} = IndexableRef{Indexable}

# from https://github.com/JuliaGPU/CUDA.jl/pull/1644
# function atomic_load(ptr::LLVMPtr{T}, order, scope::System=System()) where T
#     if order == Acq_Rel() || order == Release()
#         assert(false)
#     end
#     if compute_capability() >= sv"7.0"
#         if order == Relaxed()
#             val = __load(ptr, Relaxed(), scope)
#             return val
#         end
#         if order == Seq_Cst()
#             atomic_thread_fence(Seq_Cst(), scope)
#         end
#         val = __load(ptr, Acquire(), scope)
#         return val
#     else
#         if order == Seq_Cst()
#             atomic_thread_fence(Seq_Cst(), scope)
#         end
#         val = __load_volatile(ptr)
#         if order == Relaxed()
#             return val
#         end
#         atomic_thread_fence(order, scope)
#         return val
#     end
# end

function Atomix.get(ref::CuIndexableRef, order)
    ptr = Atomix.pointer(ref)
    return UnsafeAtomics.load(ptr, UnsafeAtomics.monotonic)
end

# function atomic_store!(ptr::LLVMPtr{T}, val::T, order, scope::System=System()) where T
#     if order == Acq_Rel() || order == Consume() || order == Acquire()
#         assert(false)
#     end
#     if compute_capability() >= sv"7.0"
#         if order == Release()
#             __store!(ptr, val, Release(), scope)
#             return
#         end
#         if order == Seq_Cst()
#             atomic_thread_fence(Seq_Cst(), scope)
#         end
#         __store!(ptr, val, Relaxed(), scope)
#     else
#         if order == Seq_Cst()
#             atomic_thread_fence(Seq_Cst(), scope)
#         end
#         __store_volatile!(ptr, val)
#     end
# end
function Atomix.set!(ref::CuIndexableRef, v, order)
    ptr = Atomix.pointer(ref)
    return UnsafeAtomics.store!(ptr, v, UnsafeAtomics.monotonic)
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
    begin
        old = CUDA.atomic_cas!(ptr, expected, desired)
    end
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(ref::CuIndexableRef, op::OP, x, order) where {OP}
    x = convert(eltype(ref), x)
    ptr = Atomix.pointer(ref)
    begin
        old = if op === (+)
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
        elseif op === Atomix.right
            CUDA.atomic_xchg!(ptr, x)
        else
            return UnsafeAtomics.modify(ptr, op, x, UnsafeAtomics.monotonic)
        end
    end
    return old => op(old, x)
end

end  # module AtomixCUDAExt
