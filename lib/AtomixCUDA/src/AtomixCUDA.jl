# TODO: respect ordering
module AtomicArraysCUDA

using AtomicArrays: AtomicRef
using CUDA: CUDA, CuArray, CuDeviceArray
using UnsafeAtomics: UnsafeAtomics, Ordering

const GenericCuArray{T,N} = Union{CuArray{T,N},CuDeviceArray{T,N}}
const CuAtomicRef{T,Ptr,Data<:GenericCuArray} = AtomicRef{T,Ptr,Data}

function UnsafeAtomics.load(ref::CuAtomicRef{T}, ord::Ordering) where {T}
    error("not implemented")
end

function UnsafeAtomics.store!(ref::CuAtomicRef{T}, v, ord::Ordering) where {T}
    error("not implemented")
end

@inline function UnsafeAtomics.cas!(
    ref::CuAtomicRef{T},
    expected,
    desired,
    success_ordering::Ordering,
    failure_ordering::Ordering,
) where {T}
    ptr = ref.ptr
    expected = convert(T, expected)
    desired = convert(T, desired)
    data = ref.data
    GC.@preserve data begin
        old = CUDA.atomic_cas!(ptr, expected, desired)
    end
    return (; old = old, success = old === expected)
end

@inline function UnsafeAtomics.modify!(ref::CuAtomicRef{T}, op::OP, x, ord) where {T,OP}
    x = convert(T, x)
    ptr = ref.ptr
    data = ref.data
    GC.@preserve data begin
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
        else
            error("not implemented")
        end
    end
    return (old, op(old, x))
end

end  # module AtomicArraysCUDA
