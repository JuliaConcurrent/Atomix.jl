# Atomic operations are atomic with respect to the threads of the device that runs them,
# like the plain atomics of CUDA C, HIP, OpenCL and Kokkos. UnsafeAtomics uses the system
# scope for that in code that Julia compiles for the host, so this only narrows the scope
# of GPU code, where a system-scope atomic can be much slower or unsupported.
@inline function Atomix.get(ref, order)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        UnsafeAtomics.load(ptr, order, UnsafeAtomics.device)
    end
end

@inline function Atomix.set!(ref, v, order)
    v = Atomix.asstorable(ref, v)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        UnsafeAtomics.store!(ptr, v, order, UnsafeAtomics.device)
    end
end

@inline function Atomix.replace!(ref, expected, desired, success_ordering, failure_ordering)
    expected = Atomix.asstorable(ref, expected)
    desired = Atomix.asstorable(ref, desired)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        UnsafeAtomics.cas!(ptr, expected, desired, success_ordering, failure_ordering,
                           UnsafeAtomics.device)
    end
end

@inline function Atomix.modify!(ref, op::OP, x, ord) where {OP}
    x = Atomix.asstorable(ref, x)
    ptr = Atomix.pointer(ref)
    root = Atomix.gcroot(ref)
    GC.@preserve root begin
        UnsafeAtomics.modify!(ptr, op, x, ord, UnsafeAtomics.device)
    end
end

Atomix.asstorable(ref, v) = convert(eltype(ref), v)
