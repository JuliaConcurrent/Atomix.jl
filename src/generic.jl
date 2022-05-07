# TODO: Support Symbol ordering

@inline Atomix.get(ref) = Atomix.get(ref, seq_cst)
@inline Atomix.set!(ref, x) = Atomix.set!(ref, x, seq_cst)

@inline Atomix.modify!(ref, op::OP, x) where {OP} = Atomix.modify!(ref, op, x, seq_cst)

@inline Atomix.replace!(ref, expected, desired, order::Ordering = seq_cst) =
    Atomix.replace!(ref, expected, desired, order, order)

@inline Atomix.swap!(ref, x, order::Ordering = seq_cst) =
    first(Atomix.modify!(ref, right, x, order))
