@inline function Atomix.get(m, lens, order)
    ptr = Atomix.pointer(m, lens)
    GC.@preserve m begin
        UnsafeAtomics.load(ptr, order)
    end
end

@inline function Atomix.set!(m, lens, v, order)
    v = Atomix.asstorable(m, v)
    ptr = Atomix.pointer(m, lens)
    GC.@preserve m begin
        UnsafeAtomics.store!(ptr, v, order)
    end
end

@inline function Atomix.replace!(
    m,
    lens,
    expected,
    desired,
    success_ordering,
    failure_ordering,
)
    expected = Atomix.asstorable(m, expected)
    desired = Atomix.asstorable(m, desired)
    ptr = Atomix.pointer(m, lens)
    GC.@preserve m begin
        UnsafeAtomics.cas!(ptr, expected, desired, success_ordering, failure_ordering)
    end
end

@inline function Atomix.modify!(m, lens, op::OP, x, ord) where {OP}
    x = Atomix.asstorable(m, x)
    ptr = Atomix.pointer(m, lens)
    GC.@preserve m begin
        UnsafeAtomics.modify!(ptr, op, x, ord)
    end
end

Atomix.asstorable(m, v) = convert(eltype(m), v)
