# TODO: respect ordering
module AtomixCUDA

using Atomix
using CUDA: CUDA, CuDeviceArray

function Atomix.get(m::CuDeviceArray{T}, lens, order) where {T}
    error("not implemented")
end

function Atomix.set!(m::CuDeviceArray{T}, lens, v, order) where {T}
    error("not implemented")
end

@inline function Atomix.replace!(
    m::CuDeviceArray{T},
    lens,
    expected,
    desired,
    success_ordering,
    failure_ordering,
) where {T}
    ptr = Atomix.pointer(m, lens)
    expected = convert(T, expected)
    desired = convert(T, desired)
    begin
        old = CUDA.atomic_cas!(ptr, expected, desired)
    end
    return (; old = old, success = old === expected)
end

@inline function Atomix.modify!(m::CuDeviceArray{T}, lens, op::OP, x, order) where {T,OP}
    x = convert(T, x)
    ptr = Atomix.pointer(m, lens)
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
        else
            error("not implemented")
        end
    end
    return (old, op(old, x))
end

end  # module AtomixCUDA
