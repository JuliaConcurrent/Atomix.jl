const _JULIA_ORDERINGS =
    (:unordered, :monotonic, :acquire, :release, :acquire_release, :sequentially_consistent)

@inline function llvm_ordering_from_juila(julia_ordering::Symbol)
    julia_ordering in _JULIA_ORDERINGS || error("unknown ordering: ", julia_ordering)
    return getfield(UnsafeAtomics, julia_ordering)
end

@inline function base_ordering(order::Ordering)
    order === seq_cst && return :sequentially_consistent
    order === acq_rel && return :acquire_release
    return Symbol(string(order))
end
